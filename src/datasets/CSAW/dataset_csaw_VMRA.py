
from __future__ import annotations

import os
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


MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME = 10


@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    exam_year: int
    laterality: str
    view: str


@dataclass(frozen=True)
class CSAWPairSample:
    patient_id: str
    current_year: int
    prior_year: int
    current_views: dict[str, CSAWImageRecord]
    prior_views: dict[str, CSAWImageRecord]


class BreastCancerRiskDatasetCSAWVMRA(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
        h5_file: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years
        self.h5_path = Path(h5_file) if h5_file else self.data_dir / f"{mode}.h5"

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {self.h5_path}")

        self._h5 = None
        self._h5_pid = None

        self.data = load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup()
        self.image_data = self._load_image_data()
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range.")

        sample = self.samples[index]
        view_order = ["L_CC", "L_MLO", "R_CC", "R_MLO"]

        prior_images = torch.stack([
            self._load_image(sample.prior_views[view].filename)
            for view in view_order
        ])
        current_images = torch.stack([
            self._load_image(sample.current_views[view].filename)
            for view in view_order
        ])

        if self.transform is not None:
            prior_images = torch.stack([
                self.transform(img).squeeze(0) for img in prior_images
            ])
            current_images = torch.stack([
                self.transform(img).squeeze(0) for img in current_images
            ])

        images = torch.stack([prior_images, current_images])

        current_row = self._lookup_row(sample.current_views["R_CC"])
        prior_row = self._lookup_row(sample.prior_views["R_CC"])

        current_ttc = clean_time_to_cancer(current_row["years_to_cancer"])
        prior_ttc = clean_time_to_cancer(prior_row["years_to_cancer"])
        current_fu = clean_followup_years(current_row["years_to_last_followup"])
        prior_fu = clean_followup_years(prior_row["years_to_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years, current_ttc, current_fu
        )
        target_prior, y_mask_prior, event_time_prior, _ = build_survival_target(
            self.n_years, prior_ttc, prior_fu
        )

        time_gap = min(abs(sample.current_year - sample.prior_year), self.n_years)

        all_views = [0, 1, 0, 1]
        all_sides = [1, 1, 0, 0]
        all_times = [0, 0, 0, 0]
        seq_len = len(all_times)

        time_seq = pad_to_length(all_times, pad_token=MAX_TIME, max_length=seq_len)
        view_seq = pad_to_length(all_views, pad_token=MAX_VIEWS, max_length=seq_len)
        side_seq = pad_to_length(all_sides, pad_token=MAX_SIDES, max_length=seq_len)

        return {
            "images": images,
            "exam_mask": torch.ones(2, dtype=torch.bool),
            "view_mask": torch.ones(2, 4, dtype=torch.bool),
            "target": target,
            "target_prior": target_prior,
            "y_mask": y_mask,
            "y_mask_prior": y_mask_prior,
            "event_observed": event_observed,
            "event_times": event_time,
            "previous_event_times": event_time_prior,
            "time_gap": time_gap,
            "density": map_density(current_row["density_group"]),
            "cancer_type": map_cancer_type(current_row["x_type"]),
            "patient_id": sample.patient_id,
            "time_seq": torch.tensor(time_seq, dtype=torch.long),
            "view_seq": torch.tensor(view_seq, dtype=torch.long),
            "side_seq": torch.tensor(side_seq, dtype=torch.long),
        }

    def _build_row_lookup(self) -> dict[str, dict]:
        lookup = {}
        for row in self.data.to_dict(orient="records"):
            filename = str(row["file_name"])
            lookup[filename] = row
            lookup[Path(filename).stem] = row
        return lookup

    def _load_image_data(self) -> dict[str, dict[int, dict[str, CSAWImageRecord]]]:
        image_data = defaultdict(lambda: defaultdict(dict))

        with h5py.File(self.h5_path, "r") as h5:
            for key in h5.keys():
                row = self.row_lookup.get(key)
                if row is None:
                    row = self.row_lookup.get(f"{key}.png")
                if row is None:
                    row = self.row_lookup.get(Path(key).stem)
                if row is None:
                    continue

                patient_id = str(row["patient_id"])
                exam_year = int(row["exam_year"])
                laterality = "L" if row["laterality"] == "Left" else "R"
                view = str(row["viewposition"])
                exam_key = f"{laterality}_{view}"

                image_data[patient_id][exam_year][exam_key] = CSAWImageRecord(
                    filename=key,
                    patient_id=patient_id,
                    exam_year=exam_year,
                    laterality=laterality,
                    view=view,
                )

        return image_data

    def _build_samples(self) -> list[CSAWPairSample]:
        required_views = {"L_CC", "L_MLO", "R_CC", "R_MLO"}
        samples = []

        for patient_id, exams_by_year in self.image_data.items():
            complete_exams = [
                (year, exams_by_year[year])
                for year in sorted(exams_by_year)
                if required_views.issubset(exams_by_year[year])
            ]

            for i in range(1, len(complete_exams)):
                prior_year, prior_views = complete_exams[i - 1]
                current_year, current_views = complete_exams[i]
                samples.append(
                    CSAWPairSample(
                        patient_id=patient_id,
                        current_year=current_year,
                        prior_year=prior_year,
                        current_views=current_views,
                        prior_views=prior_views,
                    )
                )

        print(f"[INFO] Found {len(samples)} longitudinal pairs in '{self.mode}'")
        return samples

    def _load_image(self, key: str) -> torch.Tensor:
        h5 = self._get_h5()
        if key not in h5:
            raise KeyError(f"HDF5 key '{key}' not found in {self.h5_path}")

        image = np.asarray(h5[key][()])
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image for '{key}', got {image.shape}")

        image = np.ascontiguousarray(image, dtype=np.float32) / 65535.0
        return torch.from_numpy(image).unsqueeze(0)

    def _get_h5(self):
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

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        row = self.row_lookup.get(record.filename)
        if row is None:
            row = self.row_lookup.get(f"{record.filename}.png")
        if row is None:
            raise KeyError(f"No CSV row matched image {record.filename}.")
        return row

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
