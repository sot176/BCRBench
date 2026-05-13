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
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K_A
import matplotlib.pyplot as plt
from kornia.constants import Resample


try:
    from utils import RACE_TO_ID
except ImportError:  # Keeps the module importable in minimal examples/tests.
    RACE_TO_ID = {"Unknown": 0}


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


def scale_to_uint16_range(image: np.ndarray) -> np.ndarray:
    """Scale an image array to the full uint16 intensity range."""

    image = image.astype(np.float32)
    image_min = image.min()
    image_range = image.max() - image_min

    if image_range == 0:
        return np.zeros_like(image, dtype=np.float32)

    return (image - image_min) / image_range * 65535.0


# Backwards-compatible name used in older scripts.
imgunit16 = scale_to_uint16_range


class BreastCancerRiskDatasetEMBEDImgFeatAlign(Dataset):
    """Longitudinal EMBED mammography dataset.

    Each sample contains a current mammogram, its previous mammogram from the
    same patient/laterality/view, and the risk labels derived from the CSV row
    matching the current image.
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

        self.data = self._load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup(self.data)
        self.image_data = self._load_image_data()
        self.image_pairs = self._build_image_pairs()

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | np.ndarray]:
        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(f"Index {index} is outside dataset length {len(self)}.")

        previous_record, current_record = self.image_pairs[index]
        current_image = self._load_image_tensor(current_record.filename)
        previous_image = self._load_image_tensor(previous_record.filename)

        current_row = self._lookup_row(current_record)
        previous_row = self._lookup_row(previous_record)

        if self.transform is not None:
            current_image = self.transform(current_image).squeeze(0)
            previous_image = self.transform(previous_image).squeeze(0)

        current_time_to_cancer = self._clean_time_to_cancer(
            current_row["Time_to_Cancer_Years"]
        )
        previous_time_to_cancer = self._clean_time_to_cancer(
            previous_row["Time_to_Cancer_Years"]
        )

        years_last_followup = self._clean_followup_years(
            current_row["years_last_followup"]
        )
        previous_years_last_followup = self._clean_followup_years(
            previous_row["years_last_followup"]
        )

        target, y_mask, event_time, event_observed = self._build_survival_target(
            current_time_to_cancer, years_last_followup
        )
        previous_target, previous_y_mask, previous_event_time, _ = (
            self._build_survival_target(
                previous_time_to_cancer, previous_years_last_followup
            )
        )

        time_gap = min(
            abs(current_record.study_date.year - previous_record.study_date.year),
            self.n_years,
        )

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current_record.filename,
            "previous_image_id": previous_record.filename,
            "event_observed": event_observed,
            "event_times": event_time,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": previous_y_mask,
            "years_to_cancer": current_time_to_cancer - 1,
            "years_to_cancer_prior": previous_time_to_cancer - 1,
            "years_to_last_followup": years_last_followup,
            "years_to_last_followup_prior": previous_years_last_followup,
            "target": target,
            "target_prior": previous_target,
            "density": self._map_density(current_row["density"]),
            "cancer_type": self._map_cancer_type(current_row["path_severity"]),
            "race": self._map_race(current_row.get("RACE_DESC")),
            "previous_event_times": previous_event_time,
        }

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

    def _load_image_tensor(self, filename: str) -> torch.Tensor:
        image_path = self.data_dir / self.mode / filename

        try:
            with Image.open(image_path) as image:
                image_array = np.array(image)
        except OSError as exc:
            raise OSError(f"Could not load image: {image_path}") from exc

        image_tensor = torch.from_numpy(image_array).float() / 65535.0
        return image_tensor.unsqueeze(0)

    def _lookup_row(self, record: ImageRecord) -> pd.Series:
        key = (record.patient_id, record.laterality, record.view, record.study_date)

        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(
                "No CSV row matched image "
                f"{record.filename} using key {key}."
            ) from exc

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

 

if __name__ == '__main__':
    main()