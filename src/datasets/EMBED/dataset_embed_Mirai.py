"""PyTorch dataset utilities for EMBED MIRAI-style modelling.

This module contains the multi-view EMBED mammography dataset used for
breast cancer risk prediction with MIRAI-style temporal/view embeddings.

The implementation follows the same structure and conventions as the
other EMBED datasets for repository consistency and maintainability.
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
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_"
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*\.png$"
)

MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME = 10


@dataclass(frozen=True)
class ImageRecord:
    """Metadata parsed from one mammogram filename."""

    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


@dataclass(frozen=True)
class MIRAISample:
    """Represents one MIRAI multi-view sample."""

    patient_id: str
    study_date: pd.Timestamp
    views: dict[str, ImageRecord]


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


class BreastCancerRiskDatasetEMBEDMirai(Dataset):
    """MIRAI-style EMBED mammography dataset."""

    def __init__(
        self,
        csv_file: str | os.PathLike,
        image_dir: str | os.PathLike,
        mode: str,
        transforms: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        n_years: int = 5,
        race_to_id: Optional[
            dict[str, int]
        ] = None,
    ) -> None:

        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years
        self.race_to_id = race_to_id or RACE_TO_ID

        self.data = self._load_tabular_data(csv_file)

        self.row_lookup = self._build_row_lookup(
            self.data
        )

        self.image_data = self._load_image_data()

        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str | int | np.ndarray]:

        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index {index} outside dataset length "
                f"{len(self)}."
            )

        sample = self.samples[index]

        left_cc = sample.views["L_CC"]
        left_mlo = sample.views["L_MLO"]

        right_cc = sample.views["R_CC"]
        right_mlo = sample.views["R_MLO"]

        left_cc_image = self._load_image_tensor(
            left_cc.filename
        )

        left_mlo_image = self._load_image_tensor(
            left_mlo.filename
        )

        right_cc_image = self._load_image_tensor(
            right_cc.filename
        )

        right_mlo_image = self._load_image_tensor(
            right_mlo.filename
        )

        if self.transform is not None:

            left_cc_image = self.transform(
                left_cc_image
            ).squeeze(0)

            left_mlo_image = self.transform(
                left_mlo_image
            ).squeeze(0)

            right_cc_image = self.transform(
                right_cc_image
            ).squeeze(0)

            right_mlo_image = self.transform(
                right_mlo_image
            ).squeeze(0)

        images = torch.stack(
            [
                left_cc_image,
                left_mlo_image,
                right_cc_image,
                right_mlo_image,
            ]
        )

        current_row = self._lookup_row(right_cc)

        current_time_to_cancer = (
            self._clean_time_to_cancer(
                current_row["Time_to_Cancer_Years"]
            )
        )

        years_last_followup = (
            self._clean_followup_years(
                current_row["years_last_followup"]
            )
        )

        target, y_mask, event_time, event_observed = self._build_survival_target(
                current_time_to_cancer,
                years_last_followup,
            )

        all_views = [0, 1, 0, 1]
        all_sides = [1, 1, 0, 0]
        all_times = [0, 0, 0, 0]

        time_seq = pad_to_length(all_times, MAX_TIME, len(all_times))
        view_seq = pad_to_length(all_views, MAX_VIEWS, len(all_views))
        side_seq = pad_to_length(all_sides, MAX_SIDES, len(all_sides))

        return {
            "images": images,
            "target": torch.tensor(target, dtype=torch.float32),
            "y_mask": torch.tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "density": self._map_density(current_row["density"]),
            "patient_id": sample.patient_id,
            "cancer_type": self._map_cancer_type(current_row["path_severity"]),
            "time_seq": torch.tensor(time_seq, dtype=torch.long),
            "view_seq": torch.tensor(view_seq, dtype=torch.long),
            "side_seq": torch.tensor(side_seq, dtype=torch.long),
            "race": self._map_race(current_row.get("RACE_DESC")),
            "years_to_cancer": current_time_to_cancer - 1,
            "years_to_last_followup": years_last_followup,
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

    def _load_image_data(
        self,
    ) -> dict[str, dict[pd.Timestamp, dict[str, ImageRecord]]]:

        image_data = defaultdict(lambda: defaultdict(dict))
        image_folder = self.data_dir / self.mode

        if not image_folder.exists():
            raise FileNotFoundError(
                f"Image folder does not exist: {image_folder}"
            )

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
                    datetime.strptime(
                        match.group("date"),
                        "%Y-%m-%d",
                    )
                ),
            )

            full_view = f"{record.laterality}_{record.view}"

            image_data[record.patient_id][record.study_date][
                full_view
            ] = record

        return image_data


    def _build_samples(
        self,
    ) -> list[MIRAISample]:

        samples = []

        required_views = {
            "L_CC",
            "L_MLO",
            "R_CC",
            "R_MLO",
        }

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

        print(
            f"Found {len(samples)} valid "
            f"samples in '{self.mode}' dataset."
        )

        return samples

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
    