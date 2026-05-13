import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable

import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
import kornia.augmentation.container as K_C

@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


class BreastCancerRiskDatasetCSAWCCLMVNet(Dataset):
    """Longitudinal multi-view CSAW-CC mammography dataset."""
    def __init__(
        self,
        csv_file,
        image_dir,
        mode,
        transforms: Optional[Callable] = None,
        n_years: int = 5,
    ):
        self.data_dir = image_dir
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years

        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index("file_name").to_dict(orient="index")

        self.image_data = self._load_image_data()
        self.samples = self._create_longitudinal_samples()

    def _load_image_data(self):
        image_data = defaultdict(lambda: defaultdict(dict))
        base_dir = os.path.join(self.data_dir, self.mode)

        for filename in os.listdir(base_dir):
            if filename not in self.csv_data:
                continue

            meta = self.csv_data[filename]

            patient_id = str(meta["patient_id"])
            exam_year = meta["exam_year"]
            laterality = meta["laterality"]
            view = meta["viewposition"]

            laterality = "L" if laterality == "Left" else "R"
            key = f"{laterality}_{view}"

            image_data[patient_id][exam_year][key] = {
                "filename": filename,
                "years_to_cancer": meta["years_to_cancer"],
                "years_last_followup": meta["years_to_last_followup"],
                "density": meta["density_group"],
                "cancer_type": meta["x_type"],
            }

        return image_data

    def _create_longitudinal_samples(self):
        samples = []

        for patient_id, year_dict in self.image_data.items():
            years = sorted(year_dict.keys())

            if len(years) < 2:
                continue

            for i in range(1, len(years)):
                prior_year = years[i - 1]
                current_year = years[i]

                for laterality in ["L", "R"]:
                    key_cc = f"{laterality}_CC"
                    key_mlo = f"{laterality}_MLO"

                    if (
                        key_cc in year_dict[current_year]
                        and key_mlo in year_dict[current_year]
                        and key_cc in year_dict[prior_year]
                        and key_mlo in year_dict[prior_year]
                    ):
                        samples.append(
                            (patient_id, laterality, current_year, prior_year)
                        )

        print(f"[INFO] Found {len(samples)} longitudinal samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        patient_id, laterality, current_year, prior_year = self.samples[idx]

        key_cc = f"{laterality}_CC"
        key_mlo = f"{laterality}_MLO"

        cur = self.image_data[patient_id][current_year]
        prev = self.image_data[patient_id][prior_year]

        current_cc = self._load_image(cur[key_cc]["filename"])
        current_mlo = self._load_image(cur[key_mlo]["filename"])
        previous_cc = self._load_image(prev[key_cc]["filename"])
        previous_mlo = self._load_image(prev[key_mlo]["filename"])


        meta = cur[key_cc]

        time_to_cancer = self._clean_time(meta["years_to_cancer"])
        followup = self._clean_followup(meta["years_last_followup"])

        density = self._map_density(meta["density"])
        cancer_type = self._map_cancer_type(meta["cancer_type"])

        time_gap = min(abs(current_year - prior_year), self.n_years)

        target, mask, event_time, event_observed = self._build_survival_target(
            time_to_cancer, followup
        )

        if self.transform:
            current_cc = self.transform(current_cc).squeeze(0)
            current_mlo = self.transform(current_mlo).squeeze(0)
            previous_cc = self.transform(previous_cc).squeeze(0)
            previous_mlo = self.transform(previous_mlo).squeeze(0)

        return {
            "current_image_cc": current_cc,
            "current_image_mlo": current_mlo,
            "previous_image_cc": previous_cc,
            "previous_image_mlo": previous_mlo,
            "current_image_id_cc": cur[key_cc]["filename"],
            "current_image_id_mlo": cur[key_mlo]["filename"],
            "previous_image_id_cc": prev[key_cc]["filename"],
            "previous_image_id_mlo": prev[key_mlo]["filename"],

            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "y_mask": torch.tensor(mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),

            "density": density,
            "cancer_type": cancer_type,
            "patient_id": patient_id,
        }


    def _load_image(self, filename):
        path = os.path.join(self.data_dir, self.mode, filename)

        with Image.open(path) as img:
            img = np.array(img).astype(np.float32)

        img = self._normalize_uint16(img)

        tensor = torch.from_numpy(img).float()
        return tensor.unsqueeze(0)

    @staticmethod
    def _normalize_uint16(img: np.ndarray):
        img = img.astype(np.float32)
        return img / 65535.0

    def _build_survival_target(self, ttc, followup):
        target = np.zeros(self.n_years, dtype=np.int8)

        event_time = ttc - 1
        event_observed = int(event_time < self.n_years)

        if event_observed:
            target[event_time:] = 1
        else:
            event_time = min(followup, self.n_years) - 1

        mask = np.zeros(self.n_years, dtype=np.int8)
        mask[: event_time + 1] = 1

        return target, mask, event_time, event_observed

    @staticmethod
    def _clean_time(x):
        if pd.isna(x):
            return 6
        x = int(x)
        return 1 if x == 0 else x

    @staticmethod
    def _clean_followup(x):
        if pd.isna(x):
            return 1
        x = int(x)
        return 1 if x == 0 else x

    @staticmethod
    def _map_density(x):
        return torch.tensor({"A": 1, "B": 2, "C": 3}.get(x, -1))

    @staticmethod
    def _map_cancer_type(x):
        return torch.tensor({1: 1, 2: 2, 3: 3}.get(x, -1))
    
