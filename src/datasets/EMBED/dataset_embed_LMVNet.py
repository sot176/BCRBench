"""PyTorch dataset utilities for EMBED LMVNet longitudinal modelling.

This module contains the longitudinal multi-view dataset used for breast
cancer risk prediction with current/prior CC and MLO mammograms.

The implementation follows the same structure and conventions as the
EMBED image-feature-alignment dataset for repository consistency.
"""

from __future__ import annotations

import os
import re
import random
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


IMAGE_FILENAME_RE = re.compile(
    r"(?P<patient_id>\d+)_(?P<laterality>[A-Z]+)_"
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*\.png$"
)


@dataclass(frozen=True)
class ImageRecord:
    """Metadata parsed from one mammogram filename."""

    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


@dataclass(frozen=True)
class LongitudinalSample:
    """Represents one longitudinal multi-view sample."""

    patient_id: str
    laterality: str
    current_date: pd.Timestamp
    prior_date: pd.Timestamp


def scale_to_uint16_range(image: np.ndarray) -> np.ndarray:
    """Scale image intensities to uint16 range."""

    image = image.astype(np.float32)

    image_min = image.min()
    image_range = image.max() - image_min

    if image_range == 0:
        return np.zeros_like(image, dtype=np.float32)

    return (image - image_min) / image_range * 65535.0


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

        self.data = self._load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup(self.data)

        self.image_data = self._load_image_data()
        self.samples = self._build_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str | int | np.ndarray]:

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} outside dataset length {len(self)}.")

        sample = self.samples[index]

        current_cc = self.image_data[
            sample.patient_id
        ][sample.current_date][f"{sample.laterality}_CC"]

        current_mlo = self.image_data[
            sample.patient_id
        ][sample.current_date][f"{sample.laterality}_MLO"]

        previous_cc = self.image_data[
            sample.patient_id
        ][sample.prior_date][f"{sample.laterality}_CC"]

        previous_mlo = self.image_data[
            sample.patient_id
        ][sample.prior_date][f"{sample.laterality}_MLO"]

        current_image_cc = self._load_image_tensor(current_cc.filename)
        current_image_mlo = self._load_image_tensor(current_mlo.filename)

        previous_image_cc = self._load_image_tensor(previous_cc.filename)
        previous_image_mlo = self._load_image_tensor(previous_mlo.filename)

        if self.transform is not None:
            current_image_cc = self.transform(current_image_cc).squeeze(0)
            current_image_mlo = self.transform(current_image_mlo).squeeze(0)

            previous_image_cc = self.transform(previous_image_cc).squeeze(0)
            previous_image_mlo = self.transform(previous_image_mlo).squeeze(0)

        if self.mode == "train":
            images = torch.stack(
                [
                    current_image_cc,
                    current_image_mlo,
                    previous_image_cc,
                    previous_image_mlo,
                ]
            )

            if torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[-1])

            (
                current_image_cc,
                current_image_mlo,
                previous_image_cc,
                previous_image_mlo,
            ) = images

        current_row = self._lookup_row(current_cc)

        current_time_to_cancer = self._clean_time_to_cancer(
            current_row["Time_to_Cancer_Years"]
        )

        years_last_followup = self._clean_followup_years(
            current_row["years_last_followup"]
        )

        target, y_mask, event_time, event_observed = self._build_survival_target(
                current_time_to_cancer,
                years_last_followup,
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
            "current_image_cc_id": current_cc.filename,
            "current_image_mlo_id": current_mlo.filename,
            "previous_image_cc_id": previous_cc.filename,
            "previous_image_mlo_id": previous_mlo.filename,
            "patient_id": sample.patient_id,
            "laterality": sample.laterality,
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "y_mask": torch.tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed,dtype=torch.float32,),
            "event_times": torch.tensor(event_time,dtype=torch.float32,),
            "years_to_cancer": current_time_to_cancer - 1,
            "years_to_last_followup": years_last_followup,
            "density": self._map_density(current_row["density"]),
            "cancer_type": self._map_cancer_type(current_row["path_severity"]),
            "race": self._map_race(current_row.get("RACE_DESC")),
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

    def _load_image_data(self,) -> dict[str, dict[pd.Timestamp, dict[str, ImageRecord]]]:
        image_data = defaultdict(lambda: defaultdict(dict))

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
                    datetime.strptime(
                        match.group("date"),
                        "%Y-%m-%d",
                    )
                ),
            )

            full_view = (f"{record.laterality}_{record.view}")

            image_data[record.patient_id][record.study_date][full_view] = record

        return image_data

    def _build_longitudinal_samples(
        self,
    ) -> list[LongitudinalSample]:

        samples = []

        for patient_id, date_dict in self.image_data.items():

            if len(date_dict) < 2:
                continue

            sorted_dates = sorted(date_dict.keys())

            patient_samples = []

            for idx in range(1, len(sorted_dates)):

                prior_date = sorted_dates[idx - 1]
                current_date = sorted_dates[idx]

                for laterality in ["L", "R"]:

                    cc_view = f"{laterality}_CC"
                    mlo_view = f"{laterality}_MLO"

                    if (
                        cc_view in date_dict[current_date]
                        and mlo_view in date_dict[current_date]
                        and cc_view in date_dict[prior_date]
                        and mlo_view in date_dict[prior_date]
                    ):

                        patient_samples.append(
                            LongitudinalSample(
                                patient_id=patient_id,
                                laterality=laterality,
                                current_date=current_date,
                                prior_date=prior_date,
                            )
                        )

            random.shuffle(patient_samples)
            samples.extend(patient_samples)

        random.shuffle(samples)

        print(
            f"Found {len(samples)} valid longitudinal "
            f"multi-view samples in '{self.mode}' dataset."
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
    
