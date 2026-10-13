
"""HDF5-backed CSAW-CC dataset for longitudinal LMV-Net."""
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

try:
    from .utils import (
        load_tabular_data, build_survival_target, clean_time_to_cancer,
        clean_followup_years, map_density, map_cancer_type,
    )
except ImportError:
    from utils import (
        load_tabular_data, build_survival_target, clean_time_to_cancer,
        clean_followup_years, map_density, map_cancer_type,
    )


@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    exam_year: int
    laterality: str
    view: str


@dataclass(frozen=True)
class CSAWLongitudinalSample:
    patient_id: str
    laterality: str
    current_year: int
    prior_year: int


class BreastCancerRiskDatasetCSAWCCLMVNet(Dataset):
    """Longitudinal CSAW dataset with paired current/prior CC and MLO images."""

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
        self.samples = self._build_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range.")

        sample = self.samples[index]
        current_exam = self.image_data[sample.patient_id][sample.current_year]
        prior_exam = self.image_data[sample.patient_id][sample.prior_year]
        cc_key, mlo_key = f"{sample.laterality}_CC", f"{sample.laterality}_MLO"

        cur_cc, cur_mlo = current_exam[cc_key], current_exam[mlo_key]
        pri_cc, pri_mlo = prior_exam[cc_key], prior_exam[mlo_key]

        current_image_cc = self._load_image(cur_cc.filename)
        current_image_mlo = self._load_image(cur_mlo.filename)
        previous_image_cc = self._load_image(pri_cc.filename)
        previous_image_mlo = self._load_image(pri_mlo.filename)

        if self.transform is not None:
            current_image_cc = self.transform(current_image_cc)
            current_image_mlo = self.transform(current_image_mlo)
            previous_image_cc = self.transform(previous_image_cc)
            previous_image_mlo = self.transform(previous_image_mlo)

        if self.mode == "train" and torch.rand(1).item() < 0.5:
            images = torch.stack([
                current_image_cc, current_image_mlo,
                previous_image_cc, previous_image_mlo,
            ])
            images = torch.flip(images, dims=[-1])
            current_image_cc, current_image_mlo, previous_image_cc, previous_image_mlo = images

        curr_row = self._lookup_row(cur_cc)
        curr_ttc = clean_time_to_cancer(curr_row["years_to_cancer"])
        curr_fu = clean_followup_years(curr_row["years_to_last_followup"])
        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years, curr_ttc, curr_fu
        )
        time_gap = min(abs(sample.current_year - sample.prior_year), self.n_years)

        return {
            "current_image_cc": current_image_cc,
            "current_image_mlo": current_image_mlo,
            "previous_image_cc": previous_image_cc,
            "previous_image_mlo": previous_image_mlo,
            "patient_id": sample.patient_id,
            "current_image_id_cc": cur_cc.filename,
            "current_image_id_mlo": cur_mlo.filename,
            "previous_image_id_cc": pri_cc.filename,
            "previous_image_id_mlo": pri_mlo.filename,
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "years_to_cancer": curr_ttc - 1,
            "years_to_last_followup": curr_fu,
            "density": map_density(curr_row["density_group"]),
            "cancer_type": map_cancer_type(curr_row["x_type"]),
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

    def _load_image(self, key: str) -> torch.Tensor:
        """Read a uint16 image as a normalized float tensor [1, H, W]."""
        h5 = self._get_h5()
        if key not in h5:
            raise KeyError(f"Image key '{key}' not found in {self.h5_path}")

        arr = np.ascontiguousarray(h5[key][()])
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image for '{key}', got shape {arr.shape}")

        image = torch.from_numpy(arr).float().div_(65535.0).unsqueeze(0)
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

    def _build_row_lookup(self) -> dict[str, dict]:
        return self.data.set_index("file_name").to_dict(orient="index")

    def _load_image_data(self):
        """Build patient/exam/view indexing from the HDF5 keys and CSV."""
        image_data = defaultdict(lambda: defaultdict(dict))

        with h5py.File(self.h5_path, "r") as h5:
            keys = list(h5.keys())

        print(f"[dataset] mode={self.mode}: reading {len(keys)} image keys from {self.h5_path}")

        for key in keys:
            # Support CSV file_name values both with and without .png.
            row = self.row_lookup.get(key)
            if row is None:
                row = self.row_lookup.get(f"{key}.png")
            if row is None:
                continue

            patient_id = str(row["patient_id"])
            exam_year = int(row["exam_year"])
            laterality = "L" if row["laterality"] == "Left" else "R"
            view = row["viewposition"]
            exam_key = f"{laterality}_{view}"

            image_data[patient_id][exam_year][exam_key] = CSAWImageRecord(
                filename=key, patient_id=patient_id, exam_year=exam_year,
                laterality=laterality, view=view,
            )

        return image_data

    def _build_longitudinal_samples(self) -> list[CSAWLongitudinalSample]:
        samples = []
        for patient_id, exams_by_year in self.image_data.items():
            years = sorted(exams_by_year)
            for i in range(1, len(years)):
                prior_year, current_year = years[i - 1], years[i]

                for laterality in ("L", "R"):
                    cc, mlo = f"{laterality}_CC", f"{laterality}_MLO"
                    prior, current = exams_by_year[prior_year], exams_by_year[current_year]

                    if all(k in prior and k in current for k in (cc, mlo)):
                        samples.append(CSAWLongitudinalSample(
                            patient_id, laterality, current_year, prior_year
                        ))

        random.shuffle(samples)
        print(f"Found {len(samples)} CSAW longitudinal samples in '{self.mode}'")
        return samples

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        row = self.row_lookup.get(record.filename)
        if row is None:
            row = self.row_lookup.get(f"{record.filename}.png")
        if row is None:
            raise KeyError(f"No CSV row matched image {record.filename}.")
        return row
