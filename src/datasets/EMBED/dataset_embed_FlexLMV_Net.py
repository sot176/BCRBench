from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional

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

try:
    from .utils import RACE_TO_ID
except ImportError:
    RACE_TO_ID = {"Unknown": 0}


IMAGE_FILENAME_RE = re.compile(
    r"(?P<patient_id>\d+)_(?P<laterality>[RL])_"
    r"(?P<view>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*$"
)


def _new_view_dict():
    """Picklable nested dictionary factory for DataLoader workers."""
    return defaultdict(dict)


@dataclass(frozen=True)
class ImageRecord:
    filename: str  # HDF5 key, without .png
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


@dataclass(frozen=True)
class LongitudinalSample:
    patient_id: str
    laterality: str
    current_date: pd.Timestamp
    prior_dates: tuple[pd.Timestamp, ...]


class BreastCancerRiskDatasetEMBEDFlexLMVNet(Dataset):
    """
    Multiple-prior longitudinal multi-view EMBED dataset backed by HDF5.

    Expected HDF5 files:
        <image_dir>/<mode>.h5

    Expected HDF5 keys:
        <patient_id>_<laterality>_<view>_<YYYY-MM-DD>_<index>

    Each screening examination must contain the required CC and MLO views
    for the corresponding breast laterality.

    Prior-image ordering:
        - Variable priors: newest prior first.
        - Fixed n_priors in train/validation: chronological order,
          matching the original implementation.
    """

    def __init__(
        self,
        csv_file: str | os.PathLike,
        image_dir: str | os.PathLike,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
        max_priors: int = 2,
        min_priors: int = 0,
        n_priors: Optional[int] = None,
        fixed_cohort: bool = False,
        h5_file: str | os.PathLike | None = None,
        race_to_id: Optional[dict[str, int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years
        self.max_priors = max_priors
        self.min_priors = min_priors
        self.n_priors = n_priors
        self.fixed_cohort = fixed_cohort
        self.race_to_id = race_to_id or RACE_TO_ID

        if self.max_priors < 0 or self.min_priors < 0:
            raise ValueError("max_priors and min_priors must be non-negative.")

        if self.min_priors > self.max_priors:
            raise ValueError("min_priors cannot exceed max_priors.")

        if self.n_priors is not None:
            if self.n_priors < 0:
                raise ValueError("n_priors must be non-negative or None.")
            if self.n_priors > self.max_priors:
                raise ValueError(
                    f"n_priors={self.n_priors} exceeds max_priors="
                    f"{self.max_priors}."
                )

        if h5_file is None:
            h5_file = self.data_dir / f"{self.mode}.h5"

        self.h5_path = Path(h5_file)

        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f"HDF5 file for mode '{self.mode}' not found: {self.h5_path}"
            )

        # Open the HDF5 file lazily in each process/worker.
        self._h5 = None
        self._h5_pid = None

        self.data = load_tabular_data(csv_file)
        self.row_lookup = build_row_lookup(self.data)

        self.image_data = self._load_image_data()
        self.longitudinal_samples = self._create_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.longitudinal_samples)

    # ------------------------------------------------------------------
    # HDF5 access
    # ------------------------------------------------------------------

    def _get_h5(self) -> h5py.File:
        """Return a read-only HDF5 handle specific to the current process."""
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
        """
        Read a single image from HDF5.

        Assumes mammograms are stored as uint16-like values in [0, 65535].
        Returns a float tensor with shape [1, H, W].
        """
        h5 = self._get_h5()

        if key not in h5:
            raise KeyError(
                f"HDF5 key '{key}' was not found in '{self.h5_path}'."
            )

        array = h5[key][()]

        # Handle stored images with either [H, W] or [1, H, W] shape.
        array = np.asarray(array)

        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]

        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D mammogram for key '{key}', "
                f"but received shape {array.shape}."
            )

        array = np.ascontiguousarray(array)
        tensor = torch.from_numpy(array).to(torch.float32).div_(65535.0)
        tensor = tensor.unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        # Keep the image in [C, H, W] format.
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(
                f"Transform must return [H, W] or [C, H, W]; "
                f"received {tuple(tensor.shape)} for '{key}'."
            )

        return tensor

    def __getstate__(self):
        """Ensure HDF5 handles are not pickled into DataLoader workers."""
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

    # ------------------------------------------------------------------
    # Image index
    # ------------------------------------------------------------------

    def _load_image_data(self):
        """
        Build:
            patient_id -> study_date -> view_key -> ImageRecord

        Image keys are read from the HDF5 file rather than listing PNGs.
        """
        image_data = defaultdict(_new_view_dict)

        with h5py.File(self.h5_path, "r") as h5:
            filenames = list(h5.keys())

        print(
            f"[dataset] mode={self.mode}: indexing {len(filenames)} "
            f"HDF5 keys from {self.h5_path}"
        )

        unmatched = 0

        for filename in filenames:
            match = IMAGE_FILENAME_RE.match(filename)

            if match is None:
                unmatched += 1
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

            image_data[record.patient_id][record.study_date][
                view_key
            ] = record

        if unmatched:
            print(
                f"[dataset] skipped {unmatched} HDF5 keys that did not "
                "match the expected mammogram naming convention."
            )

        print(
            f"[dataset] indexed {sum(len(d) for d in image_data.values())} "
            "patient examinations."
        )

        return image_data

    # ------------------------------------------------------------------
    # Longitudinal sample construction
    # ------------------------------------------------------------------

    def _create_longitudinal_samples(self):
        """
        Construct samples using the original multiple-prior logic.

        Train/validation:
            n_priors=None:
                Use between min_priors and max_priors recent prior exams.
            n_priors=int:
                Use sliding windows with exactly n_priors prior exams.

        Test:
            fixed_cohort=True:
                For n_priors > 0, include only current exams with at least
                max_priors available previous exams. Then select n_priors.
                For n_priors == 0, include all valid current exams.
            fixed_cohort=False:
                Use up to max_priors most recent exams, subject to min_priors.
        """
        samples = []

        for patient_id, date_dict in self.image_data.items():
            sorted_dates = sorted(date_dict.keys())

            for laterality in ("L", "R"):
                cc_view = f"{laterality}_CC"
                mlo_view = f"{laterality}_MLO"

                # Only exams with both views of the same breast.
                valid_dates = [
                    date
                    for date in sorted_dates
                    if cc_view in date_dict[date]
                    and mlo_view in date_dict[date]
                ]

                if self.mode == "test":
                    if self.fixed_cohort:
                        if self.n_priors == 0:
                            for current_date in valid_dates:
                                samples.append(
                                    LongitudinalSample(
                                        patient_id=patient_id,
                                        laterality=laterality,
                                        current_date=current_date,
                                        prior_dates=(),
                                    )
                                )
                            continue

                        # Preserve a common evaluation cohort defined by
                        # availability of max_priors, even when evaluating
                        # a smaller n_priors value.
                        if len(valid_dates) < self.max_priors + 1:
                            continue

                        for current_idx, current_date in enumerate(valid_dates):
                            available_priors = valid_dates[:current_idx][::-1]

                            if len(available_priors) < self.max_priors:
                                continue

                            selected_priors = available_priors[: self.n_priors]

                            samples.append(
                                LongitudinalSample(
                                    patient_id=patient_id,
                                    laterality=laterality,
                                    current_date=current_date,
                                    prior_dates=tuple(selected_priors),
                                )
                            )

                    else:
                        for current_idx, current_date in enumerate(valid_dates):
                            available_priors = valid_dates[:current_idx][::-1]

                            if len(available_priors) < self.min_priors:
                                continue

                            selected_priors = available_priors[: self.max_priors]

                            samples.append(
                                LongitudinalSample(
                                    patient_id=patient_id,
                                    laterality=laterality,
                                    current_date=current_date,
                                    prior_dates=tuple(selected_priors),
                                )
                            )

                else:
                    if self.n_priors is None:
                        # Variable history length.
                        for current_idx, current_date in enumerate(valid_dates):
                            available_priors = valid_dates[:current_idx][::-1]

                            if len(available_priors) < self.min_priors:
                                continue

                            selected_priors = available_priors[: self.max_priors]

                            samples.append(
                                LongitudinalSample(
                                    patient_id=patient_id,
                                    laterality=laterality,
                                    current_date=current_date,
                                    prior_dates=tuple(selected_priors),
                                )
                            )

                    else:
                        # Fixed-length sliding windows. Preserve chronological
                        # order for consistency with the original dataset.
                        window_size = self.n_priors + 1

                        if len(valid_dates) < window_size:
                            continue

                        for start_idx in range(
                            len(valid_dates) - window_size + 1
                        ):
                            window = valid_dates[
                                start_idx : start_idx + window_size
                            ]

                            samples.append(
                                LongitudinalSample(
                                    patient_id=patient_id,
                                    laterality=laterality,
                                    current_date=window[-1],
                                    prior_dates=tuple(window[:-1]),
                                )
                            )

        # Avoid shuffling validation/test samples unnecessarily.
        if self.mode == "train":
            if self.n_priors is None:
                rng = random.Random(self.seed if hasattr(self, "seed") else None)
                rng.shuffle(samples)
            else:
                random.shuffle(samples)

        prior_counts = defaultdict(int)
        for sample in samples:
            prior_counts[len(sample.prior_dates)] += 1

        print(f"[dataset] mode={self.mode}: prior-count distribution:")
        for count in sorted(prior_counts):
            print(f"  {count} prior(s): {prior_counts[count]} samples")

        print(
            f"[dataset] found {len(samples)} longitudinal samples "
            f"for mode='{self.mode}'."
        )

        return samples

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _lookup_row(self, record: ImageRecord):
        """Look up the CSV row matching a specific image record."""
        key = (
            record.patient_id,
            record.laterality,
            record.view,
            record.study_date,
        )

        try:
            return self.row_lookup[key]
        except KeyError as exc:
            raise KeyError(
                "No matching metadata row for image "
                f"{record.filename!r}. Expected lookup key: {key!r}. "
                "Check that patient_id, laterality, view, and study date "
                "are normalized consistently in the CSV and lookup utility."
            ) from exc

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.longitudinal_samples):
            raise IndexError(
                f"Index {index} outside dataset length {len(self)}."
            )

        sample = self.longitudinal_samples[index]

        patient_id = sample.patient_id
        laterality = sample.laterality
        current_date = sample.current_date
        prior_dates = sample.prior_dates

        current_views = self.image_data[patient_id][current_date]
        current_cc = current_views[f"{laterality}_CC"]
        current_mlo = current_views[f"{laterality}_MLO"]

        current_image_cc = self._read_image_tensor(current_cc.filename)
        current_image_mlo = self._read_image_tensor(current_mlo.filename)

        prior_images_cc = []
        prior_images_mlo = []
        time_gaps = []

        for prior_date in prior_dates:
            prior_views = self.image_data[patient_id][prior_date]
            prior_cc = prior_views[f"{laterality}_CC"]
            prior_mlo = prior_views[f"{laterality}_MLO"]

            prior_images_cc.append(
                self._read_image_tensor(prior_cc.filename)
            )
            prior_images_mlo.append(
                self._read_image_tensor(prior_mlo.filename)
            )

            gap = min(
                max(current_date.year - prior_date.year, 0),
                self.n_years,
            )
            time_gaps.append(float(gap))

        # One shared horizontal flip decision for the current exam and
        # all priors, keeping spatial orientation consistent longitudinally.
        if self.mode == "train" and torch.rand(()) < 0.5:
            current_image_cc = torch.flip(current_image_cc, dims=[-1])
            current_image_mlo = torch.flip(current_image_mlo, dims=[-1])

            prior_images_cc = [
                torch.flip(img, dims=[-1]) for img in prior_images_cc
            ]
            prior_images_mlo = [
                torch.flip(img, dims=[-1]) for img in prior_images_mlo
            ]

        time_gaps = torch.tensor(time_gaps, dtype=torch.float32)

        # Use the current CC image's metadata, matching the original code.
        current_row = self._lookup_row(current_cc)

        ttc = clean_time_to_cancer(current_row["Time_to_Cancer_Years"])
        followup = clean_followup_years(
            current_row["years_last_followup"]
        )

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            ttc,
            followup,
        )

        race = map_race(current_row.get("RACE_DESC"), self.race_to_id)

        return {
            "current_image_cc": current_image_cc,
            "current_image_mlo": current_image_mlo,
            "previous_image_cc": prior_images_cc,
            "previous_image_mlo": prior_images_mlo,
            "time_gap": time_gaps,
            "target": torch.as_tensor(target, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(
                event_observed, dtype=torch.float32
            ),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "density": map_density(current_row["density"]),
            "patient_id": patient_id,
            "cancer_type": map_cancer_type(current_row["path_severity"]),
            "race": torch.as_tensor(race, dtype=torch.long),
            "n_priors": len(prior_dates),
        }