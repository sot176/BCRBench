"""PyTorch dataset utilities for EMBED VMRA longitudinal modelling."""

from __future__ import annotations

import os
import re
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample

try:
    from utils import RACE_TO_ID
except ImportError:
    RACE_TO_ID = {"Unknown": 0}

"""
 Expected filename format:
#     <patient_id>_<laterality>_<view>_<YYYY-MM-DD>_<index>.png
#
# example:
     10093833_L_CC_2017-08-03_4505.png
     10093833_L_MLO_2017-08-03_5642.png
     10093833_R_MLO_2017-08-03_2678.png
     10093833_R_CC_2017-08-03_4123.png
     10093833_R_CC_2014-07-02_7821.png
     10093833_R_MLO_2014-07-02_4689.png
     10093833_L_MLO_2014-07-02_7897.png
     10093833_L_CC_2014-07-02_6798.png


CSV FORMAT REQUIREMENT:
Each row corresponds to exactly ONE IMAGE.
A full screening exam consists of 4 rows:
    - L_CC
    - L_MLO
    - R_CC
    - R_MLO
All rows share the same patient_id and study_date_anon.
"""

IMAGE_FILENAME_RE = re.compile(
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*\.png$"
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


def scale_to_uint16_range(image: np.ndarray) -> np.ndarray:
    """Normalize mammogram using fixed physical intensity scale."""

    image = image.astype(np.float32)
    image = image / 65535.0
    return  image 


def pad_to_length(
    arr: list[int],
    pad_token: int,
    max_length: int,
) -> np.ndarray:
    """Pad sequence to fixed length."""

    arr = arr[-max_length:]
    return np.array([pad_token] * (max_length - len(arr)) + arr)


class BreastCancerRiskDatasetEMBEDVMRA(Dataset):

    def __init__(
        self,
        csv_file: str | os.PathLike,
        image_dir: str | os.PathLike,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
        race_to_id: Optional[dict[str, int]] = None,
    ) -> None:

        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years
        self.race_to_id = race_to_id or RACE_TO_ID

        self.data = self._load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup(self.data)

        self.image_data = self._load_image_data()

        self.samples = self._build_longitudinal_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        def load(rec: ImageRecord):
            img = self._load_image_tensor(rec.filename)
            if self.transform:
                img = self.transform(img).squeeze(0)
            return img

        curr = torch.stack([
            load(self.image_data[sample.patient_id][sample.current_date][v])
            for v in ["L_CC", "L_MLO", "R_CC", "R_MLO"]
        ])

        prior = torch.stack([
            load(self.image_data[sample.patient_id][sample.prior_date][v])
            for v in ["L_CC", "L_MLO", "R_CC", "R_MLO"]
        ])

        images = torch.stack([prior, curr])  # [2,4,3,H,W]

        row = self._lookup_row(
            self.image_data[sample.patient_id][sample.current_date]["R_CC"]
        )

        ttc = self._clean_time_to_cancer(row["Time_to_Cancer_Years"])
        follow = self._clean_followup_years(row["years_last_followup"])

        target, y_mask, event_time, event_observed = self._build_survival_target(
            ttc, follow
        )

        all_views = [0, 1, 0, 1]
        all_sides = [1, 1, 0, 0]
        all_times = [0, 0, 0, 0]

        time_seq = pad_to_length(all_times, MAX_TIME, len(all_times))
        view_seq = pad_to_length(all_views, MAX_VIEWS, len(all_views))
        side_seq = pad_to_length(all_sides, MAX_SIDES, len(all_sides))
        
        exam_mask = torch.ones(2, dtype=torch.bool) 
        view_mask = torch.ones(2, 4, dtype=torch.bool)
        
        return {
            'images': images,
            'exam_mask': exam_mask,
            'view_mask': view_mask,
            'target': target,
            'y_mask': y_mask,
            'event_observed': event_observed,
            'event_times': torch.tensor(event_time, dtype=torch.float32),
            "patient_id": sample.patient_id,
            'time_seq': torch.tensor(time_seq, dtype=torch.long),
            'view_seq': torch.tensor(view_seq, dtype=torch.long),
            'side_seq': torch.tensor(side_seq, dtype=torch.long),
            "density": self._map_density(row["density"]),
            "cancer_type": self._map_cancer_type(row["path_severity"]),
            "race": self._map_race(row.get("RACE_DESC")),
        }

   
    def _load_image_data(self):
        data = defaultdict(lambda: defaultdict(dict))
        path = self.data_dir / self.mode

        for f in os.listdir(path):
            m = IMAGE_FILENAME_RE.match(f)
            if not m:
                continue

            pid, lat, view, date = m.group(
                "patient_id", "laterality", "view", "date"
            )
            date = datetime.strptime(date, "%Y-%m-%d")

            record = ImageRecord(
                filename=f,
                patient_id=pid,
                laterality=lat,
                view=view,
                study_date=pd.Timestamp(date),
            )

            key = f"{lat}_{view}"
            data[pid][record.study_date][key] = record

        return data

    def _build_longitudinal_samples(self):

        samples = []
        required = {"L_CC", "L_MLO", "R_CC", "R_MLO"}

        for pid, date_dict in self.image_data.items():

            dates = sorted(date_dict.keys())
            if len(dates) < 2:
                continue

            for i in range(1, len(dates)):

                prior, curr = dates[i - 1], dates[i]

                if (
                    required.issubset(date_dict[prior].keys())
                    and required.issubset(date_dict[curr].keys())
                ):
                    samples.append(
                        LongitudinalSample(
                            patient_id=pid,
                            current_date=curr,
                            prior_date=prior,
                        )
                    )

        random.shuffle(samples)
        print(f"Found {len(samples)} longitudinal samples")
        return samples


    def _load_image_tensor(self, fname):
        path = self.data_dir / self.mode / fname
        img = np.array(Image.open(path))
        t = torch.from_numpy(img).float() / 65535.0
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t

    def _lookup_row(self, record: ImageRecord) -> pd.Series:
        key = (record.patient_id, record.laterality, record.view, record.study_date)

        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(
                "No CSV row matched image "
                f"{record.filename} using key {key}."
            ) from exc
        
    @staticmethod
    def _load_tabular_data(csv_file: str | os.PathLike) -> pd.DataFrame:
        data = pd.read_csv(csv_file, low_memory=False)
        data["study_date_anon"] = pd.to_datetime(
            data["study_date_anon"], format="%Y-%m-%d"
        )
        return data

    @staticmethod
    def _build_row_lookup(data: pd.DataFrame) -> dict[tuple[str, str, str, pd.Timestamp], pd.Series]:
        lookup: dict[tuple[str, str, str, pd.Timestamp], pd.Series] = {}

        for _, row in data.iterrows():
            key = (
                str(row["patient_id"]),
                str(row["ImageLateralityFinal"]),
                str(row["view"]),
                row["study_date_anon"],
            )
            lookup[key] = row

        return lookup

    def _map_density(self, value: object) -> torch.Tensor:
        value = -1 if pd.isna(value) else int(value)
        return torch.tensor(value if value in {1, 2, 3, 4} else -1, dtype=torch.long)

    @staticmethod
    def _map_cancer_type(value: object) -> torch.Tensor:
        value = -1 if pd.isna(value) else int(value)
        return torch.tensor(value if value in set(range(7)) else -1, dtype=torch.long)

    def _map_race(self, value: object) -> torch.Tensor:
        if not isinstance(value, str) or value.strip() == "":
            race = "Unknown"
        else:
            race = value.strip()

        race_id = self.race_to_id.get(race, self.race_to_id["Unknown"])
        return torch.tensor(race_id, dtype=torch.long)

    @staticmethod
    def _clean_time_to_cancer(value: object) -> int:
        if pd.isna(value):
            return 6

        value = int(value)
        return 1 if value == 0 else value

    @staticmethod
    def _clean_followup_years(value: object) -> int:
        if pd.isna(value):
            return 1

        value = int(value)
        return 1 if value == 0 else value

    def _build_survival_target(
        self,
        time_to_cancer: int,
        years_last_followup: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        target = np.zeros(self.n_years, dtype=np.int8)
        event_time = time_to_cancer - 1
        event_observed = int(event_time < self.n_years)

        if event_observed:
            target[event_time:] = 1
        else:
            event_time = min(years_last_followup, self.n_years) - 1

        y_mask = np.zeros(self.n_years, dtype=np.int8)
        y_mask[: event_time + 1] = 1

        return target, y_mask, event_time, event_observed
