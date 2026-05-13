from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .datasetutils import (
    load_tabular_data,
    load_image_tensor,
    map_density,
    map_cancer_type,
    map_race,
    clean_time_to_cancer,
    clean_followup_years,
    build_survival_target,
    build_row_lookup,
    pad_to_length,
)

try:
    from BreastCancerRiskBenchmark.src.datasets.EMBED.datasetutils import RACE_TO_ID
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


MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME = 10


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


@dataclass(frozen=True)
class MIRAISample:
    patient_id: str
    study_date: pd.Timestamp
    views: dict[str, ImageRecord]


class BreastCancerRiskDatasetEMBEDMirai(Dataset):
    """MIRAI-style EMBED mammography dataset."""

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
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} outside dataset length {len(self)}")

        sample = self.samples[index]

        # ---- unpack views ----
        left_cc = sample.views["L_CC"]
        left_mlo = sample.views["L_MLO"]
        right_cc = sample.views["R_CC"]
        right_mlo = sample.views["R_MLO"]

        # ---- load images ----
        images = torch.stack([
            load_image_tensor(self.data_dir / self.mode / left_cc.filename),
            load_image_tensor(self.data_dir / self.mode / left_mlo.filename),
            load_image_tensor(self.data_dir / self.mode / right_cc.filename),
            load_image_tensor(self.data_dir / self.mode / right_mlo.filename),
        ])

        # ---- optional transform ----
        if self.transform:
            images = torch.stack([self.transform(x).squeeze(0) for x in images])

        # ---- tabular lookup (use one view as anchor) ----
        current_row = self._lookup_row(right_cc)

        ttc = clean_time_to_cancer(current_row["Time_to_Cancer_Years"])
        followup = clean_followup_years(current_row["years_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            ttc,
            followup,
        )

      # ---- simple temporal encoding ----
        base_views = [0, 1, 0, 1]
        base_sides = [1, 1, 0, 0]
        base_times = [0, 0, 0, 0]

        view_seq = torch.tensor(pad_to_length(base_views, MAX_VIEWS, len(base_views)), dtype=torch.long)
        side_seq = torch.tensor(pad_to_length(base_sides, MAX_SIDES, len(base_sides)), dtype=torch.long)
        time_seq = torch.tensor(pad_to_length(base_times, MAX_TIME, len(base_times)), dtype=torch.long)

        # ---- masks ----
        exam_mask = torch.ones(2, dtype=torch.bool)
        view_mask = torch.ones(2, 4, dtype=torch.bool)


        return {
            "images": images,
            "exam_mask": exam_mask,
            "view_mask": view_mask,
            "target": torch.tensor(target, dtype=torch.float32),
            "y_mask": torch.tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "density": map_density(current_row["density"]),
            "cancer_type": map_cancer_type(current_row["path_severity"]),
            "race": map_race(current_row.get("RACE_DESC"), self.race_to_id),
            "patient_id": sample.patient_id,
            "view_seq": view_seq,
            "side_seq": side_seq,
            "time_seq": time_seq,
            "years_to_cancer": ttc - 1,
            "years_to_last_followup": followup,
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

            key = f"{record.laterality}_{record.view}"
            image_data[record.patient_id][record.study_date][key] = record

        return image_data

    def _build_samples(self):

        samples = []
        required_views = {"L_CC", "L_MLO", "R_CC", "R_MLO"}

        for patient_id, date_dict in self.image_data.items():
            for study_date, views in date_dict.items():

                if required_views.issubset(views.keys()):
                    samples.append(
                        MIRAISample(
                            patient_id=patient_id,
                            study_date=study_date,
                            views=views,
                        )
                    )

        print(f"Found {len(samples)} valid MIRAI samples in '{self.mode}' dataset.")
        return samples

    def _lookup_row(self, record: ImageRecord):
        key = (
            record.patient_id,
            record.laterality,
            record.view,
            record.study_date,
        )
        return self.row_lookup[key]