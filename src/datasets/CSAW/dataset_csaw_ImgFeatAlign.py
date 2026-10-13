
"""HDF5-backed CSAW-CC dataset for longitudinal image feature alignment."""
from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    load_tabular_data,
    build_survival_target,
    clean_time_to_cancer,
    clean_followup_years,
    map_density,
    map_cancer_type,
    pad_to_length,
)

@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    exam_year: int
    laterality: str
    view: str


class BreastCancerRiskDatasetCSAWCCImgFeatAlign(Dataset):
    """Longitudinal CSAW-CC dataset backed by a split-specific HDF5 file."""

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
        h5_file: str | Path | None = None,
    ) -> None:
        self.data_dir, self.mode = Path(image_dir), mode
        self.transform, self.n_years = transforms, n_years
        self.h5_path = Path(h5_file) if h5_file else self.data_dir / f"{mode}.h5"

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {self.h5_path}")

        self._h5, self._h5_pid = None, None
        self.data = load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup()
        self.image_data = self._load_image_data()
        self.image_pairs = self._build_image_pairs()

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(f"Index {index} out of range.")

        prev_record, curr_record = self.image_pairs[index]
        current_image = self._read_image_tensor(curr_record.filename)
        previous_image = self._read_image_tensor(prev_record.filename)

        if self.transform is not None:
            current_image = self.transform(current_image)
            previous_image = self.transform(previous_image)

        # Apply the same horizontal flip to both time points.
        if self.mode == "train" and torch.rand(1).item() < 0.5:
            current_image = torch.flip(current_image, dims=[-1])
            previous_image = torch.flip(previous_image, dims=[-1])

        curr_row, prev_row = self._lookup_row(curr_record), self._lookup_row(prev_record)
        curr_ttc = clean_time_to_cancer(curr_row["years_to_cancer"])
        prev_ttc = clean_time_to_cancer(prev_row["years_to_cancer"])
        curr_fu = clean_followup_years(curr_row["years_to_last_followup"])
        prev_fu = clean_followup_years(prev_row["years_to_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years, curr_ttc, curr_fu
        )
        prev_target, prev_y_mask, prev_event_time, _ = build_survival_target(
            self.n_years, prev_ttc, prev_fu
        )
        time_gap = min(abs(curr_record.exam_year - prev_record.exam_year), self.n_years)

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": curr_record.filename,
            "previous_image_id": prev_record.filename,
            "event_observed": torch.as_tensor(event_observed, dtype=torch.float32),
            "event_times": torch.as_tensor(event_time, dtype=torch.float32),
            "previous_event_times": torch.as_tensor(prev_event_time, dtype=torch.float32),
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "y_mask_prior": torch.as_tensor(prev_y_mask, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "target_prior": torch.as_tensor(prev_target, dtype=torch.float32),
            "years_to_cancer": curr_ttc - 1,
            "years_to_cancer_prior": prev_ttc - 1,
            "years_to_last_followup": curr_fu,
            "years_to_last_followup_prior": prev_fu,
            "density": map_density(curr_row["density_group"]),
            "cancer_type": map_cancer_type(curr_row["x_type"]),
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

    def _read_image_tensor(self, key: str) -> torch.Tensor:
        """Read a uint16 grayscale image as a normalized [1, H, W] tensor."""
        h5 = self._get_h5()
        if key not in h5:
            raise KeyError(f"Image key '{key}' not found in {self.h5_path}")

        arr = np.ascontiguousarray(h5[key][()])
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image for '{key}', got {arr.shape}")

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

    def _build_row_lookup(self) -> dict[str, dict]:
        return self.data.set_index("file_name").to_dict(orient="index")

    def _load_image_data(self) -> dict[tuple[str, str, str], list[CSAWImageRecord]]:
        image_data = defaultdict(list)

        with h5py.File(self.h5_path, "r") as h5:
            keys = list(h5.keys())

        print(f"[dataset] mode={self.mode}: reading {len(keys)} image keys from {self.h5_path}")

        for key in keys:
            # HDF5 keys may omit the original .png extension.
            row = self.row_lookup.get(key)
            if row is None:
                row = self.row_lookup.get(f"{key}.png")
            if row is None:
                continue

            patient_id = str(row["patient_id"])
            exam_year = int(row["exam_year"])
            laterality = "L" if row["laterality"] == "Left" else "R"
            view = row["viewposition"]

            record = CSAWImageRecord(
                filename=key,
                patient_id=patient_id,
                exam_year=exam_year,
                laterality=laterality,
                view=view,
            )
            image_data[(patient_id, laterality, view)].append(record)

        return image_data

    def _build_image_pairs(self) -> list[tuple[CSAWImageRecord, CSAWImageRecord]]:
        pairs = []
        for records in self.image_data.values():
            records = sorted(records, key=lambda record: (record.exam_year, record.filename))
            pairs.extend(zip(records[:-1], records[1:]))

        print(f"Found {len(pairs)} image pairs in {self.mode}")
        return pairs

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        row = self.row_lookup.get(record.filename)
        if row is None:
            row = self.row_lookup.get(f"{record.filename}.png")
        if row is None:
            raise KeyError(f"No CSV row matched image {record.filename}.")
        return row
