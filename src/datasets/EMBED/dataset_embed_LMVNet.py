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

from BreastCancerRiskBenchmark.src.datasets.EMBED.utils import (
    load_tabular_data,
    load_image_tensor,
    map_density,
    map_cancer_type,
    map_race,
    clean_time_to_cancer,
    clean_followup_years,
    build_survival_target,
    build_row_lookup,
)

try:
    from BreastCancerRiskBenchmark.src.datasets.EMBED.utils import RACE_TO_ID
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
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_"
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*\.png$"
)


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
    laterality: str
    current_date: pd.Timestamp
    prior_date: pd.Timestamp


class BreastCancerRiskDatasetEMBEDLMVNet(Dataset):
    """Longitudinal multi-view EMBED mammography dataset."""

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

        # ---- shared utilities ----
        self.data = load_tabular_data(csv_file)
        self.row_lookup = build_row_lookup(self.data)

        # ---- dataset-specific indexing ----
        self.image_data = self._load_image_data()
        self.samples = self._build_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} outside dataset length {len(self)}.")

        sample = self.samples[index]

        # ---- fetch image records ----
        current_cc = self.image_data[sample.patient_id][sample.current_date][f"{sample.laterality}_CC"]
        current_mlo = self.image_data[sample.patient_id][sample.current_date][f"{sample.laterality}_MLO"]

        previous_cc = self.image_data[sample.patient_id][sample.prior_date][f"{sample.laterality}_CC"]
        previous_mlo = self.image_data[sample.patient_id][sample.prior_date][f"{sample.laterality}_MLO"]

        # ---- load images ----
        current_image_cc = load_image_tensor(self.data_dir / self.mode / current_cc.filename)
        current_image_mlo = load_image_tensor(self.data_dir / self.mode / current_mlo.filename)

        previous_image_cc = load_image_tensor(self.data_dir / self.mode / previous_cc.filename)
        previous_image_mlo = load_image_tensor(self.data_dir / self.mode / previous_mlo.filename)

        # ---- transforms ----
        if self.transform:
            current_image_cc = self.transform(current_image_cc).squeeze(0)
            current_image_mlo = self.transform(current_image_mlo).squeeze(0)
            previous_image_cc = self.transform(previous_image_cc).squeeze(0)
            previous_image_mlo = self.transform(previous_image_mlo).squeeze(0)

            # grouped tensor for augmentation consistency
            images = torch.stack(
                [
                    current_image_cc,
                    current_image_mlo,
                    previous_image_cc,
                    previous_image_mlo,
                ]
            )

            if self.mode == "train" and torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[-1])

            (
                current_image_cc,
                current_image_mlo,
                previous_image_cc,
                previous_image_mlo,
            ) = images

        # ---- tabular lookup ----
        current_row = self._lookup_row(current_cc)

        ttc = clean_time_to_cancer(current_row["Time_to_Cancer_Years"])
        followup = clean_followup_years(current_row["years_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            ttc,
            followup,
        )

        time_gap = min(
            abs(sample.current_date.year - sample.prior_date.year),
            self.n_years,
        )

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
            "density": map_density(current_row["density"]),
            "cancer_type": map_cancer_type(current_row["path_severity"]),
            "race": map_race(current_row.get("RACE_DESC"), self.race_to_id),
        }

    def _load_image_data(self):
        image_data = defaultdict(lambda: defaultdict(dict))

        image_folder = self.data_dir / self.mode

        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        for image_path in image_folder.iterdir():

            match = IMAGE_FILENAME_RE.match(image_path.name)
            if not match:
                continue

            record = ImageRecord(
                filename=image_path.name,
                patient_id=match.group("patient_id"),
                laterality=match.group("laterality"),
                view=match.group("view"),
                study_date=pd.Timestamp(
                    datetime.strptime(match.group("date"), "%Y-%m-%d")
                ),
            )

            view_key = f"{record.laterality}_{record.view}"
            image_data[record.patient_id][record.study_date][view_key] = record

        return image_data

    def _build_longitudinal_samples(self):

        samples = []

        for patient_id, date_dict in self.image_data.items():

            if len(date_dict) < 2:
                continue

            sorted_dates = sorted(date_dict.keys())

            for idx in range(1, len(sorted_dates)):

                prior_date = sorted_dates[idx - 1]
                current_date = sorted_dates[idx]

                for laterality in ["L", "R"]:

                    cc = f"{laterality}_CC"
                    mlo = f"{laterality}_MLO"

                    if (
                        cc in date_dict[current_date]
                        and mlo in date_dict[current_date]
                        and cc in date_dict[prior_date]
                        and mlo in date_dict[prior_date]
                    ):
                        samples.append(
                            LongitudinalSample(
                                patient_id=patient_id,
                                laterality=laterality,
                                current_date=current_date,
                                prior_date=prior_date,
                            )
                        )

        random.shuffle(samples)

        print(
            f"Found {len(samples)} valid longitudinal "
            f"multi-view samples in '{self.mode}' dataset."
        )

        return samples

    def _lookup_row(self, record: ImageRecord):
        key = (
            record.patient_id,
            record.laterality,
            record.view,
            record.study_date,
        )
        return self.row_lookup[key]