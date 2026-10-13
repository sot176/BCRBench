
"""HDF5-backed EMBED dataset for longitudinal VMRA."""
from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (
    load_tabular_data,
    map_density,
    map_cancer_type,
    map_race,
    clean_time_to_cancer,
    clean_followup_years,
    build_survival_target,
    build_row_lookup,
    pad_to_length,
)

try:
    from .utils import RACE_TO_ID
except ImportError:  # Keeps the module importable in minimal examples/tests.
    RACE_TO_ID = {"Unknown": 0}


IMAGE_FILENAME_RE = re.compile(
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_"
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*$"
)

MAX_VIEWS, MAX_SIDES, MAX_TIME = 2, 2, 10


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


@dataclass(frozen=True)
class LongitudinalSample:
    patient_id: str
    current_date: pd.Timestamp
    prior_date: pd.Timestamp


class BreastCancerRiskDatasetEMBEDVMRA(Dataset):
    """Longitudinal EMBED dataset backed by a split-specific HDF5 file."""

    def __init__(
        self,
        csv_file: str | os.PathLike,
        image_dir: str | os.PathLike,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
        race_to_id: Optional[dict[str, int]] = None,
        h5_file: str | os.PathLike | None = None,
    ) -> None:
        self.data_dir, self.mode = Path(image_dir), mode
        self.transform, self.n_years = transforms, n_years
        self.race_to_id = race_to_id or RACE_TO_ID
        self.h5_path = Path(h5_file) if h5_file else self.data_dir / f"{mode}.h5"

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {self.h5_path}")

        self._h5, self._h5_pid = None, None
        self.data = load_tabular_data(csv_file)
        self.row_lookup = build_row_lookup(self.data)
        self.image_data = self._load_image_data()
        self.samples = self._build_longitudinal_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} outside dataset length {len(self)}")

        sample = self.samples[idx]
        view_names = ["L_CC", "L_MLO", "R_CC", "R_MLO"]
        prior_records = [
            self.image_data[sample.patient_id][sample.prior_date][v]
            for v in view_names
        ]
        current_records = [
            self.image_data[sample.patient_id][sample.current_date][v]
            for v in view_names
        ]

        prior = torch.stack([self._load_image(r.filename) for r in prior_records])
        curr = torch.stack([self._load_image(r.filename) for r in current_records])
        images = torch.stack([prior, curr])  # [2, 4, H, W]

        row = self._lookup_row(self.image_data[sample.patient_id][sample.current_date]["R_CC"])
        ttc = clean_time_to_cancer(row["Time_to_Cancer_Years"])
        follow = clean_followup_years(row["years_last_followup"])
        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years, ttc, follow
        )

        base_views, base_sides, base_times = [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]
        view_seq = torch.tensor(pad_to_length(base_views, MAX_VIEWS, len(base_views)), dtype=torch.long)
        side_seq = torch.tensor(pad_to_length(base_sides, MAX_SIDES, len(base_sides)), dtype=torch.long)
        time_seq = torch.tensor(pad_to_length(base_times, MAX_TIME, len(base_times)), dtype=torch.long)

        return {
            "images": images,
            "exam_mask": torch.ones(2, dtype=torch.bool),
            "view_mask": torch.ones(2, 4, dtype=torch.bool),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "patient_id": sample.patient_id,
            "time_seq": time_seq,
            "view_seq": view_seq,
            "side_seq": side_seq,
            "density": map_density(row["density"]),
            "cancer_type": map_cancer_type(row["path_severity"]),
            "race": map_race(row.get("RACE_DESC"), self.race_to_id),
        }

    def _get_h5(self) -> h5py.File:
        """Open a read-only HDF5 handle for the current worker process."""
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5, self._h5_pid = h5py.File(self.h5_path, "r"), pid
        return self._h5

    def _load_image(self, key: str) -> torch.Tensor:
        """Load one uint16 grayscale image as a normalized [H, W] tensor."""
        h5 = self._get_h5()
        if key not in h5:
            raise KeyError(f"Image key '{key}' not found in {self.h5_path}")

        arr = np.ascontiguousarray(h5[key][()])
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D image for '{key}', got shape {arr.shape}")

        image = torch.from_numpy(arr).float().div_(65535.0)
        if self.transform is not None:
            image = self.transform(image.unsqueeze(0)).squeeze(0)
        return image

    def __getstate__(self):
        """Avoid serializing HDF5 handles when using spawned workers."""
        state = self.__dict__.copy()
        state["_h5"], state["_h5_pid"] = None, None
        return state

    def __del__(self):
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except Exception:
            pass

    def _load_image_data(self):
        """Index HDF5 image keys by patient, date, and view."""
        data = defaultdict(lambda: defaultdict(dict))
        with h5py.File(self.h5_path, "r") as h5:
            filenames = list(h5.keys())

        print(f"[dataset] mode={self.mode}: reading {len(filenames)} images from {self.h5_path}")

        for filename in filenames:
            match = IMAGE_FILENAME_RE.match(filename)
            if match is None:
                continue

            record = ImageRecord(
                filename=filename,
                patient_id=match.group("patient_id"),
                laterality=match.group("laterality"),
                view=match.group("view"),
                study_date=pd.Timestamp(datetime.strptime(match.group("date"), "%Y-%m-%d")),
            )
            data[record.patient_id][record.study_date][
                f"{record.laterality}_{record.view}"
            ] = record

        return data

    def _build_longitudinal_samples(self):
        samples, required = [], {"L_CC", "L_MLO", "R_CC", "R_MLO"}

        for pid, date_dict in self.image_data.items():
            dates = sorted(date_dict.keys())
            for i in range(1, len(dates)):
                prior, curr = dates[i - 1], dates[i]
                if required.issubset(date_dict[prior]) and required.issubset(date_dict[curr]):
                    samples.append(LongitudinalSample(pid, curr, prior))

        random.shuffle(samples)
        print(f"Found {len(samples)} longitudinal samples in '{self.mode}' dataset.")
        return samples

    def _lookup_row(self, record: ImageRecord):
        key = (record.patient_id, record.laterality, record.view, record.study_date)
        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(f"No CSV row matched image '{record.filename}' using key {key}") from exc
