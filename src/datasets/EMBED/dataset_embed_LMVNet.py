from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (
    load_tabular_data,
    map_density,
    map_cancer_type,
    map_race,
    clean_time_to_cancer,
    clean_followup_years,
    build_survival_target,
    build_row_lookup,
)
from ..augmentation_utils import grouped_hflip

try:
    from .utils import RACE_TO_ID
except ImportError:
    RACE_TO_ID = {"Unknown": 0}


"""
 Expected HDF5 key format (PNG filename stem, no extension):
#     <patient_id>_<laterality>_<view>_<YYYY-MM-DD>_<index>
#
# Images for a split live in a single HDF5 file, ``<image_dir>/<mode>.h5``
# (e.g. ``train.h5`` / ``val.h5`` / ``test.h5``), where each key is the original
# PNG filename without the ``.png`` extension.
#
# example keys:
     10093833_L_CC_2017-08-03_4505
     10093833_L_MLO_2017-08-03_5642
     10093833_R_MLO_2017-08-03_2678
     10093833_R_CC_2017-08-03_4123
     10093833_R_CC_2014-07-02_7821
     10093833_R_MLO_2014-07-02_4689
     10093833_L_MLO_2014-07-02_7897
     10093833_L_CC_2014-07-02_6798


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
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*$"
)


def _new_view_dict():
    # Module-level factory so the nested defaultdict stays picklable
    # (required by DataLoader workers using the spawn start method).
    return defaultdict(dict)


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
        h5_file: str | os.PathLike | None = None,
    ) -> None:

        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years
        self.race_to_id = race_to_id or RACE_TO_ID

        # ---- HDF5 backend ----
        # Images for this split live in a single HDF5 file whose keys are the
        # PNG filename stems, e.g. "10093833_L_CC_2017-08-03_4505".
        if h5_file is None:
            h5_file = self.data_dir / f"{self.mode}.h5"
        self.h5_path = Path(h5_file)
        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f"HDF5 file for mode '{self.mode}' does not exist: "
                f"{self.h5_path}"
            )
        # Per-process HDF5 handle opened lazily inside DataLoader workers.
        self._h5 = None
        self._h5_pid = None

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
        current_image_cc = self._read_image_tensor(current_cc.filename)
        current_image_mlo = self._read_image_tensor(current_mlo.filename)

        previous_image_cc = self._read_image_tensor(previous_cc.filename)
        previous_image_mlo = self._read_image_tensor(previous_mlo.filename)

        # ---- transforms ----
        if self.transform:
            current_image_cc = self.transform(current_image_cc).squeeze(0)
            current_image_mlo = self.transform(current_image_mlo).squeeze(0)
            previous_image_cc = self.transform(previous_image_cc).squeeze(0)
            previous_image_mlo = self.transform(previous_image_mlo).squeeze(0)

        # ---- horizontal flip ----
        # Applied to all views of the sample together (single random decision) so
        # current/prior and CC/MLO stay geometrically consistent. Kept independent
        # of `self.transform` because the flips' group-consistency is essential and
        # must not depend on whether augmentation transforms are supplied.
        if self.mode == "train":
            (
                current_image_cc,
                current_image_mlo,
                previous_image_cc,
                previous_image_mlo,
            ) = grouped_hflip(
                [
                    current_image_cc,
                    current_image_mlo,
                    previous_image_cc,
                    previous_image_mlo,
                ]
            )

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

    def _get_h5(self) -> h5py.File:
        """Return a per-process read-only HDF5 handle (reopened per worker)."""
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5 = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
        return self._h5

    def _read_image_tensor(self, key: str) -> torch.Tensor:
        """Load a single image from the HDF5 file as a float tensor."""
        arr = self._get_h5()[key][()]
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float() / 65535.0
        return tensor.unsqueeze(0)

    def __getstate__(self):
        # h5py.File handles are not picklable (e.g. Windows/spawn workers);
        # drop it and let each worker lazily reopen its own handle.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except Exception:
            pass

    def _load_image_data(self):
        image_data = defaultdict(_new_view_dict)

        with h5py.File(self.h5_path, "r") as h5:
            filenames = list(h5.keys())

        print(
            f"[dataset] mode={self.mode}: reading {len(filenames)} "
            f"images from {self.h5_path}"
        )

        for filename in filenames:

            match = IMAGE_FILENAME_RE.match(filename)
            if not match:
                continue

            record = ImageRecord(
                filename=filename,
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