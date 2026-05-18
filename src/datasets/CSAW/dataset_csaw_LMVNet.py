from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader


import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
import kornia.augmentation.container as K_C


from utils import (
    load_tabular_data,
    load_image_tensor,
    build_survival_target,
    clean_time_to_cancer,
    clean_followup_years,
    map_density,
    map_cancer_type,
)


@dataclass(frozen=True)
class CSAWLongitudinalSample:
    patient_id: str
    laterality: str
    current_year: int
    prior_year: int


class BreastCancerRiskDatasetCSAWCCLMVNet(Dataset):
    """
    Longitudinal multi-view CSAW-CC mammography dataset.
    Each sample contains:
        - current CC + MLO
        - prior CC + MLO
        - survival labels + metadata
    """

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        mode: str,
        transforms: Optional[Callable] = None,
        n_years: int = 5,
    ) -> None:

        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years

        self.data = load_tabular_data(csv_file)
        self.csv_lookup = self.data.set_index("file_name").to_dict(orient="index")

        self.image_data = self._load_image_data()
        self.samples = self._build_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} outside dataset length {len(self)}.")

        sample = self.samples[index]

        key_cc = f"{sample.laterality}_CC"
        key_mlo = f"{sample.laterality}_MLO"

        current_exam = self.image_data[sample.patient_id][sample.current_year]
        prior_exam = self.image_data[sample.patient_id][sample.prior_year]

        current_image_cc = load_image_tensor(self.data_dir / self.mode / current_exam[key_cc]["filename"])
        current_image_mlo = load_image_tensor(self.data_dir / self.mode / current_exam[key_mlo]["filename"])

        previous_image_cc = load_image_tensor(self.data_dir / self.mode / prior_exam[key_cc]["filename"])
        previous_image_mlo = load_image_tensor(self.data_dir / self.mode / prior_exam[key_mlo]["filename"])

        if self.transform:
            current_image_cc = self.transform(current_image_cc).squeeze(0)
            current_image_mlo = self.transform(current_image_mlo).squeeze(0)
            previous_image_cc = self.transform(previous_image_cc).squeeze(0)
            previous_image_mlo = self.transform(previous_image_mlo).squeeze(0)

            images = torch.stack([
                current_image_cc,
                current_image_mlo,
                previous_image_cc,
                previous_image_mlo,
            ])

            if self.mode == "train" and torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[-1])

            current_image_cc, current_image_mlo, previous_image_cc, previous_image_mlo = images

        meta = current_exam[key_cc]

        time_to_cancer = clean_time_to_cancer(meta["years_to_cancer"])
        followup = clean_followup_years(meta["years_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years, time_to_cancer, followup
        )

        time_gap = min(abs(sample.current_year - sample.prior_year), self.n_years)

        return {
            "current_image_cc": current_image_cc,
            "current_image_mlo": current_image_mlo,
            "previous_image_cc": previous_image_cc,
            "previous_image_mlo": previous_image_mlo,
            "patient_id": sample.patient_id,
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "y_mask": torch.tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "density": map_density(meta["density"]),
            "cancer_type": map_cancer_type(meta["cancer_type"]),
        }

    def _load_image_data(self):

        image_data = defaultdict(lambda: defaultdict(dict))
        image_folder = self.data_dir / self.mode

        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        for image_path in image_folder.iterdir():

            filename = image_path.name

            if filename not in self.csv_lookup:
                continue

            meta = self.csv_lookup[filename]

            patient_id = str(meta["patient_id"])
            exam_year = int(meta["exam_year"])
            laterality = "L" if meta["laterality"] == "Left" else "R"
            view = meta["viewposition"]

            view_key = f"{laterality}_{view}"

            image_data[patient_id][exam_year][view_key] = {
                "filename": filename,
                "years_to_cancer": meta["years_to_cancer"],
                "years_last_followup": meta["years_to_last_followup"],
                "density": meta["density_group"],
                "cancer_type": meta["x_type"],
            }

        return image_data

    def _build_longitudinal_samples(self):

        samples = []

        for patient_id, year_dict in self.image_data.items():

            if len(year_dict) < 2:
                continue

            sorted_years = sorted(year_dict.keys())

            for idx in range(1, len(sorted_years)):

                prior_year = sorted_years[idx - 1]
                current_year = sorted_years[idx]

                for laterality in ["L", "R"]:

                    cc = f"{laterality}_CC"
                    mlo = f"{laterality}_MLO"

                    if (
                        cc in year_dict[current_year]
                        and mlo in year_dict[current_year]
                        and cc in year_dict[prior_year]
                        and mlo in year_dict[prior_year]
                    ):
                        samples.append(
                            CSAWLongitudinalSample(
                                patient_id=patient_id,
                                laterality=laterality,
                                current_year=current_year,
                                prior_year=prior_year,
                            )
                        )

        random.shuffle(samples)

        print(f"Found {len(samples)} valid longitudinal multi-view CSAW samples in '{self.mode}'.")

        return samples
    
