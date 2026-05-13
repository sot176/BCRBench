
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset 
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable
 
@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


class BreastCancerRiskDatasetCSAWCC_ImgFeatAlign(Dataset):
    """Longitudinal CSAW-CC mammography dataset.

    Each sample contains a current mammogram, its previous mammogram from the
    same patient/laterality/view, and the risk labels derived from the CSV row
    matching the current image.
    """

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
        self.image_pairs = self._build_image_pairs()


    def _load_image_data(self):
        image_data = defaultdict(list)
        base_dir = os.path.join(self.data_dir, self.mode)

        for filename in os.listdir(base_dir):
            if filename not in self.csv_data:
                continue

            meta = self.csv_data[filename]

            record = CSAWImageRecord(
                filename=filename,
                patient_id=str(meta["patient_id"]),
                laterality=meta["laterality"],
                view=meta["viewposition"],
                study_date=pd.Timestamp(meta["exam_year"]),
            )

            key = (record.patient_id, record.laterality, record.view)
            image_data[key].append(record)

        for key in image_data:
            image_data[key].sort(key=lambda x: x.study_date)

        return image_data

    def _build_image_pairs(self):
        pairs = []

        for records in self.image_data.values():
            for i in range(len(records) - 1):
                pairs.append((records[i], records[i + 1]))

        return pairs


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        current, previous = self.image_pairs[idx]

        current_img = self._load_image(current.filename)
        previous_img = self._load_image(previous.filename)

        cur_meta = self.csv_data[current.filename]
        prev_meta = self.csv_data[previous.filename]

        time_to_cancer = self._clean_time(cur_meta["years_to_cancer"])
        time_to_cancer_prev = self._clean_time(prev_meta["years_to_cancer"])

        followup = self._clean_followup(cur_meta["years_to_last_followup"])
        followup_prev = self._clean_followup(prev_meta["years_to_last_followup"])

        time_gap = min(
            abs(current.study_date.year - previous.study_date.year),
            self.n_years,
        )

        target, mask, event_time, event_observed = self._build_survival_target(
            time_to_cancer, followup
        )

        target_prev, mask_prev, event_time_prev, _ = self._build_survival_target(
            time_to_cancer_prev, followup_prev
        )

        if self.transform:
            current_img = self.transform(current_img).squeeze(0)
            previous_img = self.transform(previous_img).squeeze(0)

        return {
            "current_image": current_img,
            "previous_image": previous_img,
            "current_image_id": current.filename,
            "previous_image_id": previous.filename,
            "event_observed": event_observed,
            "event_times": event_time,
            "time_gap": time_gap,
            "y_mask": mask,
            "y_mask_prior": mask_prev,
            "target": target,
            "target_prior": target_prev,
            "density": self._map_density(cur_meta["density_group"]),
            "cancer_type": self._map_cancer_type(cur_meta["x_type"]),
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


    def _build_survival_target(self, time_to_cancer, followup):
        target = np.zeros(self.n_years, dtype=np.int8)

        event_time = time_to_cancer - 1
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
        mapping = {"A": 1, "B": 2, "C": 3}
        return torch.tensor(mapping.get(x, -1), dtype=torch.long)

    @staticmethod
    def _map_cancer_type(x):
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        return torch.tensor(mapping.get(x, -1), dtype=torch.long)

 