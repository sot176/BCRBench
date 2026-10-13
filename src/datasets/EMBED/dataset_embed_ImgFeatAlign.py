
"""HDF5-backed EMBED dataset for longitudinal image feature alignment."""
from __future__ import annotations

import os
import re
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .utils import (
        load_tabular_data, map_density, map_cancer_type, map_race,
        clean_time_to_cancer, clean_followup_years, build_survival_target,
        build_row_lookup, RACE_TO_ID,
    )
except ImportError:
    from utils import (
        load_tabular_data, map_density, map_cancer_type, map_race,
        clean_time_to_cancer, clean_followup_years, build_survival_target,
        build_row_lookup,
    )
    try:
        from utils import RACE_TO_ID
    except ImportError:
        RACE_TO_ID = {"Unknown": 0}


IMAGE_FILENAME_RE = re.compile(
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*$"
)


@dataclass(frozen=True)
class ImageRecord:
    """Metadata parsed from an HDF5 image key."""
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


class BreastCancerRiskDatasetEMBEDImgFeatAlign(Dataset):
    """Longitudinal EMBED dataset with previous/current images per patient, side, and view."""

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
            raise FileNotFoundError(f"HDF5 file for mode '{mode}' does not exist: {self.h5_path}")

        self._h5, self._h5_pid = None, None
        self.data = load_tabular_data(csv_file)
        self.row_lookup = build_row_lookup(self.data)
        self.image_data = self._load_image_data()
        self.image_pairs = self._build_image_pairs()
        print(f"[dataset] mode={mode}: found {len(self.image_pairs)} longitudinal image pairs")

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(f"Index {index} outside dataset length {len(self)}.")

        previous_record, current_record = self.image_pairs[index]
        current_image = self._read_image_tensor(current_record.filename)
        previous_image = self._read_image_tensor(previous_record.filename)

        if self.transform is not None:
            current_image, previous_image = self.transform(current_image), self.transform(previous_image)

        if self.mode == "train" and torch.rand(1).item() < 0.5:
            current_image, previous_image = torch.flip(current_image, [-1]), torch.flip(previous_image, [-1])

        curr_row, prev_row = self._lookup_row(current_record), self._lookup_row(previous_record)
        curr_ttc = clean_time_to_cancer(curr_row["Time_to_Cancer_Years"])
        prev_ttc = clean_time_to_cancer(prev_row["Time_to_Cancer_Years"])
        curr_fu = clean_followup_years(curr_row["years_last_followup"])
        prev_fu = clean_followup_years(prev_row["years_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(self.n_years, curr_ttc, curr_fu)
        prev_target, prev_y_mask, prev_event_time, _ = build_survival_target(self.n_years, prev_ttc, prev_fu)
        time_gap = min(abs(current_record.study_date.year - previous_record.study_date.year), self.n_years)

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current_record.filename,
            "previous_image_id": previous_record.filename,
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "previous_event_times": torch.tensor(prev_event_time, dtype=torch.float32),
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "y_mask_prior": torch.as_tensor(prev_y_mask, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "target_prior": torch.as_tensor(prev_target, dtype=torch.float32),
            "years_to_cancer": curr_ttc - 1,
            "years_to_cancer_prior": prev_ttc - 1,
            "years_to_last_followup": curr_fu,
            "years_to_last_followup_prior": prev_fu,
            "density": map_density(curr_row["density"]),
            "cancer_type": map_cancer_type(curr_row["path_severity"]),
            "race": map_race(curr_row.get("RACE_DESC"), self.race_to_id),
        }

    def _get_h5(self) -> h5py.File:
        """Open one read-only HDF5 handle per worker process."""
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5, self._h5_pid = h5py.File(self.h5_path, "r"), pid
        return self._h5

    def _read_image_tensor(self, key: str) -> torch.Tensor:
        """Read a uint16 image as a normalized float tensor [1, H, W]."""
        h5 = self._get_h5()
        if key not in h5:
            raise KeyError(f"Image key '{key}' not found in {self.h5_path}")
        arr = np.ascontiguousarray(h5[key][()])
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image for '{key}', got shape {arr.shape}")
        return torch.from_numpy(arr).float().div_(65535.0).unsqueeze(0)

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
        """Read image-key metadata and group by patient, side, and view."""
        image_data = defaultdict(list)
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
            image_data[(record.patient_id, record.laterality, record.view)].append(record)
        return image_data

    def _build_image_pairs(self):
        """Pair consecutive exams for each patient, side, and view."""
        pairs = []
        for records in self.image_data.values():
            records = sorted(records, key=lambda r: (r.study_date, r.filename))
            pairs.extend(zip(records[:-1], records[1:]))
        random.shuffle(pairs)
        return pairs

    def _lookup_row(self, record: ImageRecord) -> pd.Series:
        key = (record.patient_id, record.laterality, record.view, record.study_date)
        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(f"No CSV row matched image '{record.filename}' using key {key}.") from exc
