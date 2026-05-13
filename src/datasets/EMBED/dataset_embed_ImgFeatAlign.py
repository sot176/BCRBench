"""PyTorch dataset utilities for EMBED mammography risk modelling.

This module contains the image-pair dataset used for longitudinal breast
cancer risk prediction. It intentionally avoids hard-coded local paths so it
can be imported from training scripts, notebooks, and published repositories.
"""

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

from BreastCancerRiskBenchmark.src.datasets.EMBED.datasetutils import (
    load_tabular_data,
    load_image_tensor,
    map_density,
    map_cancer_type,
    map_race,
    clean_time_to_cancer,
    clean_followup_years,
    build_survival_target,
    build_row_lookup
)

try:
    from BreastCancerRiskBenchmark.src.datasets.EMBED.datasetutils import RACE_TO_ID
except ImportError:  # Keeps the module importable in minimal examples/tests.
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
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_(?P<view>[A-Z]+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_.*\.png$"
)


@dataclass(frozen=True)
class ImageRecord:
    """Metadata parsed from one mammogram filename."""

    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp



class BreastCancerRiskDatasetEMBEDImgFeatAlign(Dataset):
    """Longitudinal EMBED mammography dataset.

    Each sample contains a current mammogram and the previous mammogram
    for the same patient/laterality/view, along with survival targets.
    """

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

        # ---- dataset-specific structure ----
        self.image_data = self._load_image_data()
        self.image_pairs = self._build_image_pairs()

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):

        if index >= len(self.image_pairs) or index < 0:
            raise IndexError(f"Index {index} out of range.")

        prev_record, curr_record = self.image_pairs[index]

        # ---- images ----
        current_image = load_image_tensor(
            self.data_dir / self.mode / curr_record.filename
        )
        previous_image = load_image_tensor(
            self.data_dir / self.mode / prev_record.filename
        )

        if self.transform:
            current_image = self.transform(current_image).squeeze(0)
            previous_image = self.transform(previous_image).squeeze(0)

        # ---- rows ----
        curr_row = self._lookup_row(curr_record)
        prev_row = self._lookup_row(prev_record)

        # ---- survival inputs ----
        curr_ttc = clean_time_to_cancer(curr_row["Time_to_Cancer_Years"])
        prev_ttc = clean_time_to_cancer(prev_row["Time_to_Cancer_Years"])

        curr_fu = clean_followup_years(curr_row["years_last_followup"])
        prev_fu = clean_followup_years(prev_row["years_last_followup"])

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

        # ---- temporal gap ----
        time_gap = min(
            abs(curr_record.study_date.year - prev_record.study_date.year),
            self.n_years,
        )

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": curr_record.filename,
            "previous_image_id": prev_record.filename,
            "event_observed": event_observed,
            "event_times": event_time,
            "previous_event_times": prev_event_time,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": prev_y_mask,
            "target": target,
            "target_prior": prev_target,
            "years_to_cancer": curr_ttc - 1,
            "years_to_cancer_prior": prev_ttc - 1,
            "years_to_last_followup": curr_fu,
            "years_to_last_followup_prior": prev_fu,
            "density": map_density(curr_row["density"]),
            "cancer_type": map_cancer_type(curr_row["path_severity"]),
            "race": map_race(curr_row.get("RACE_DESC"), self.race_to_id),
        }


    @staticmethod
    def _load_tabular_data(csv_file: str | os.PathLike) -> pd.DataFrame:
        data = pd.read_csv(csv_file, low_memory=False)
        data["study_date_anon"] = pd.to_datetime(
            data["study_date_anon"], format="%Y-%m-%d"
        )
        return data


    def _load_image_data(self) -> dict[tuple[str, str, str], list[ImageRecord]]:
        image_data: dict[tuple[str, str, str], list[ImageRecord]] = defaultdict(list)
        image_folder = self.data_dir / self.mode

        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        for image_path in image_folder.iterdir():
            match = IMAGE_FILENAME_RE.match(image_path.name)
            if match is None:
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
            image_data[(record.patient_id, record.laterality, record.view)].append(record)

        return image_data

    def _build_image_pairs(self) -> list[tuple[ImageRecord, ImageRecord]]:
        pairs: list[tuple[ImageRecord, ImageRecord]] = []

        for records in self.image_data.values():
            records = sorted(records, key=lambda record: record.study_date)
            pairs.extend(zip(records[:-1], records[1:]))

        return pairs

    def _lookup_row(self, record: ImageRecord) -> pd.Series:
        key = (record.patient_id, record.laterality, record.view, record.study_date)

        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(
                "No CSV row matched image "
                f"{record.filename} using key {key}."
            ) from exc


  

