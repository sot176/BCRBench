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
from torch.utils.data import Dataset

from .utils import (
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
    from utils import RACE_TO_ID
except ImportError:
    RACE_TO_ID = {"Unknown": 0}


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

        # ---- shared utilities ----
        self.data = load_tabular_data(csv_file)
        self.row_lookup = build_row_lookup(self.data)

        # ---- dataset-specific indexing ----
        self.image_data = self._load_image_data()
        self.samples = self._build_longitudinal_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):

        sample = self.samples[idx]

        def load(rec: ImageRecord):
            img = load_image_tensor(self.data_dir / self.mode / rec.filename)
            if self.transform:
                img = self.transform(img).squeeze(0)
            return img

        # ---- current exam ----
        curr = torch.stack([
            load(self.image_data[sample.patient_id][sample.current_date][v])
            for v in ["L_CC", "L_MLO", "R_CC", "R_MLO"]
        ])

        # ---- prior exam ----
        prior = torch.stack([
            load(self.image_data[sample.patient_id][sample.prior_date][v])
            for v in ["L_CC", "L_MLO", "R_CC", "R_MLO"]
        ])

        images = torch.stack([prior, curr])  # [2, 4, C, H, W]

        # ---- anchor row ----
        row = self._lookup_row(
            self.image_data[sample.patient_id][sample.current_date]["R_CC"]
        )

        # ---- survival ----
        ttc = clean_time_to_cancer(row["Time_to_Cancer_Years"])
        follow = clean_followup_years(row["years_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            ttc,
            follow,
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
            "patient_id": sample.patient_id,
            "time_seq": time_seq,
            "view_seq": view_seq,
            "side_seq": side_seq,
            "density": map_density(row["density"]),
            "cancer_type": map_cancer_type(row["path_severity"]),
            "race": map_race(row.get("RACE_DESC"), self.race_to_id),
        }

    def _load_image_data(self):
        data = defaultdict(lambda: defaultdict(dict))
        path = self.data_dir / self.mode

        for f in os.listdir(path):

            m = IMAGE_FILENAME_RE.match(f)
            if not m:
                continue

            pid = m.group("patient_id")
            lat = m.group("laterality")
            view = m.group("view")
            date = datetime.strptime(m.group("date"), "%Y-%m-%d")

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
                    required.issubset(date_dict[prior])
                    and required.issubset(date_dict[curr])
                ):
                    samples.append(
                        LongitudinalSample(
                            patient_id=pid,
                            current_date=curr,
                            prior_date=prior,
                        )
                    )

        random.shuffle(samples)
        print(f"Found {len(samples)} longitudinal samples: {len(samples)}")

        return samples

    def _lookup_row(self, record: ImageRecord):
        key = (
            record.patient_id,
            record.laterality,
            record.view,
            record.study_date,
        )
        return self.row_lookup[key]