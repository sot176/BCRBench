
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
        load_tabular_data,
        map_density,
        map_cancer_type,
        map_race,
        clean_time_to_cancer,
        clean_followup_years,
        build_survival_target,
        build_row_lookup,
        RACE_TO_ID,
    )
except ImportError:
    from utils import (
        load_tabular_data,
        map_density,
        map_cancer_type,
        map_race,
        clean_time_to_cancer,
        clean_followup_years,
        build_survival_target,
        build_row_lookup,
    )

    try:
        from utils import RACE_TO_ID
    except ImportError:
        RACE_TO_ID = {"Unknown": 0}


# HDF5 keys have no .png extension:
# <patient_id>_<laterality>_<view>_<YYYY-MM-DD>_<index>
IMAGE_FILENAME_RE = re.compile(
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_"
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*$"
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
    """Longitudinal EMBED dataset backed by a split-specific HDF5 file.

    Each sample contains a previous and current image for the same
    patient, laterality, and view.
    """

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
        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years
        self.race_to_id = race_to_id or RACE_TO_ID

        # HDF5 backend
        if h5_file is None:
            h5_file = self.data_dir / f"{self.mode}.h5"

        self.h5_path = Path(h5_file)

        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f"HDF5 file for mode '{self.mode}' does not exist: "
                f"{self.h5_path}"
            )

        # Each process/DataLoader worker opens its own HDF5 handle.
        self._h5 = None
        self._h5_pid = None

        # Shared tabular utilities
        self.data = load_tabular_data(csv_file)
        self.row_lookup = build_row_lookup(self.data)

        # Index image metadata and build longitudinal pairs.
        self.image_data = self._load_image_data()
        self.image_pairs = self._build_image_pairs()

        print(
            f"[dataset] mode={self.mode}: "
            f"found {len(self.image_pairs)} longitudinal image pairs"
        )

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(
                f"Index {index} outside dataset length {len(self)}."
            )

        previous_record, current_record = self.image_pairs[index]

        # Load images from HDF5.
        current_image = self._read_image_tensor(current_record.filename)
        previous_image = self._read_image_tensor(previous_record.filename)

        # Keep a consistent [C, H, W] shape, with or without transforms.
        if self.transform is not None:
            current_image = self.transform(current_image)
            previous_image = self.transform(previous_image)

        # Apply the same random horizontal flip to both time points.
        if self.mode == "train" and torch.rand(1).item() < 0.5:
            current_image = torch.flip(current_image, dims=[-1])
            previous_image = torch.flip(previous_image, dims=[-1])

        # Tabular metadata.
        curr_row = self._lookup_row(current_record)
        prev_row = self._lookup_row(previous_record)

        curr_ttc = clean_time_to_cancer(
            curr_row["Time_to_Cancer_Years"]
        )
        prev_ttc = clean_time_to_cancer(
            prev_row["Time_to_Cancer_Years"]
        )

        curr_fu = clean_followup_years(
            curr_row["years_last_followup"]
        )
        prev_fu = clean_followup_years(
            prev_row["years_last_followup"]
        )

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            curr_ttc,
            curr_fu,
        )

        prev_target, prev_y_mask, prev_event_time, _ = build_survival_target(
            self.n_years,
            prev_ttc,
            prev_fu,
        )

        time_gap = min(
            abs(
                current_record.study_date.year
                - previous_record.study_date.year
            ),
            self.n_years,
        )

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current_record.filename,
            "previous_image_id": previous_record.filename,
            "event_observed": torch.tensor(
                event_observed, dtype=torch.float32
            ),
            "event_times": torch.tensor(
                event_time, dtype=torch.float32
            ),
            "previous_event_times": torch.tensor(
                prev_event_time, dtype=torch.float32
            ),
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "y_mask_prior": torch.as_tensor(
                prev_y_mask, dtype=torch.float32
            ),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "target_prior": torch.as_tensor(
                prev_target, dtype=torch.float32
            ),
            "years_to_cancer": curr_ttc - 1,
            "years_to_cancer_prior": prev_ttc - 1,
            "years_to_last_followup": curr_fu,
            "years_to_last_followup_prior": prev_fu,
            "density": map_density(curr_row["density"]),
            "cancer_type": map_cancer_type(
                curr_row["path_severity"]
            ),
            "race": map_race(
                curr_row.get("RACE_DESC"), self.race_to_id
            ),
        }

    def _get_h5(self) -> h5py.File:
        """Return a read-only HDF5 handle for the current process."""
        pid = os.getpid()

        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass

            self._h5 = h5py.File(self.h5_path, "r")
            self._h5_pid = pid

        return self._h5

    def _read_image_tensor(self, key: str) -> torch.Tensor:
        """Read a uint16 mammogram and return a float [1, H, W] tensor."""
        h5 = self._get_h5()

        if key not in h5:
            raise KeyError(
                f"Image key '{key}' not found in {self.h5_path}"
            )

        arr = h5[key][()]
        arr = np.ascontiguousarray(arr)

        tensor = torch.from_numpy(arr).float() / 65535.0

        # The HDF5 array is expected to be [H, W].
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale image for '{key}', "
                f"got shape {tuple(tensor.shape)}"
            )

        return tensor.unsqueeze(0)

    def __getstate__(self):
        """Avoid serializing HDF5 handles when workers use spawn."""
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        try:
            h5 = getattr(self, "_h5", None)
            if h5 is not None:
                h5.close()
        except Exception:
            pass

    def _load_image_data(self):
        """Read image-key metadata from the HDF5 file."""
        image_data = defaultdict(list)

        with h5py.File(self.h5_path, "r") as h5:
            filenames = list(h5.keys())

        print(
            f"[dataset] mode={self.mode}: reading {len(filenames)} "
            f"images from {self.h5_path}"
        )

        for filename in filenames:
            match = IMAGE_FILENAME_RE.match(filename)

            if match is None:
                continue

            record = ImageRecord(
                filename=filename,
                patient_id=match.group("patient_id"),
                laterality=match.group("laterality"),
                view=match.group("view"),
                study_date=pd.Timestamp(
                    datetime.strptime(
                        match.group("date"), "%Y-%m-%d"
                    )
                ),
            )

            group_key = (
                record.patient_id,
                record.laterality,
                record.view,
            )
            image_data[group_key].append(record)

        return image_data

    def _build_image_pairs(self):
        """Pair consecutive exams for each patient, side, and view."""
        pairs = []

        for records in self.image_data.values():
            records = sorted(
                records,
                key=lambda record: (
                    record.study_date,
                    record.filename,
                ),
            )

            pairs.extend(zip(records[:-1], records[1:]))

        # Preserve the original dataset's randomized sample order.
        random.shuffle(pairs)

        return pairs

    def _lookup_row(self, record: ImageRecord) -> pd.Series:
        key = (
            record.patient_id,
            record.laterality,
            record.view,
            record.study_date,
        )

        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(
                f"No CSV row matched image '{record.filename}' "
                f"using key {key}."
            ) from exc
