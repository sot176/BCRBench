
"""HDF5-backed CSAW-CC dataset for multiple-prior longitudinal risk prediction."""
from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    load_tabular_data,
    build_survival_target,
    clean_time_to_cancer,
    clean_followup_years,
    map_density,
    map_cancer_type,
)


@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    exam_year: int
    laterality: str
    view: str


@dataclass(frozen=True)
class CSAWLongitudinalSample:
    patient_id: str
    laterality: str
    current_year: int
    prior_years: tuple[int, ...]


class BreastCancerRiskDatasetCSAWFlexLMVNet(Dataset):
    """Multiple-prior longitudinal CSAW-CC dataset backed by HDF5."""

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
        max_priors: int = 2,
        min_priors: int = 0,
        n_priors: Optional[int] = None,
        fixed_cohort: bool = False,
        h5_file: str | Path | None = None,
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
        self.seed = seed

        if self.max_priors < 0 or self.min_priors < 0:
            raise ValueError("max_priors and min_priors must be non-negative.")
        if self.min_priors > self.max_priors:
            raise ValueError("min_priors cannot exceed max_priors.")
        if self.n_priors is not None:
            if not 0 <= self.n_priors <= self.max_priors:
                raise ValueError("n_priors must be between 0 and max_priors.")

        self.h5_path = (
            Path(h5_file) if h5_file is not None
            else self.data_dir / f"{self.mode}.h5"
        )
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {self.h5_path}")

        # Each DataLoader worker opens its own HDF5 handle lazily.
        self._h5 = None
        self._h5_pid = None

        self.data = load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup()
        self.image_data = self._load_image_data()
        self.samples = self._build_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # HDF5 access
    # ------------------------------------------------------------------

    def _get_h5(self) -> h5py.File:
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

    def _load_image(self, key: str) -> torch.Tensor:
        """Read a grayscale mammogram as float tensor [1, H, W]."""
        h5 = self._get_h5()

        if key not in h5:
            raise KeyError(f"Image key '{key}' not found in {self.h5_path}")

        arr = np.asarray(h5[key][()])

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale image for '{key}', got {arr.shape}"
            )

        arr = np.ascontiguousarray(arr)
        image = torch.from_numpy(arr).float().div_(65535.0).unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(
                f"Transform must return [H, W] or [C, H, W], "
                f"got {tuple(image.shape)} for '{key}'."
            )

        return image

    def __getstate__(self):
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
    # CSV lookup and image indexing
    # ------------------------------------------------------------------

    def _build_row_lookup(self) -> dict[str, dict]:
        """Index metadata by filename, with and without .png."""
        lookup = {}

        for filename, row in self.data.set_index("file_name").to_dict(
            orient="index"
        ).items():
            name = str(filename)
            lookup[name] = row
            if name.endswith(".png"):
                lookup[name[:-4]] = row
            else:
                lookup[f"{name}.png"] = row

        return lookup

    def _load_image_data(self):
        """Build patient -> exam year -> view -> image record."""
        image_data = defaultdict(lambda: defaultdict(dict))

        with h5py.File(self.h5_path, "r") as h5:
            keys = list(h5.keys())

        print(
            f"[dataset] mode={self.mode}: reading {len(keys)} image keys "
            f"from {self.h5_path}"
        )

        unmatched = 0

        for key in keys:
            row = self.row_lookup.get(key)
            if row is None:
                unmatched += 1
                continue

            patient_id = str(row["patient_id"])
            exam_year = int(row["exam_year"])

            raw_laterality = str(row["laterality"]).strip()
            if raw_laterality == "Left":
                laterality = "L"
            elif raw_laterality == "Right":
                laterality = "R"
            else:
                continue

            view = str(row["viewposition"]).strip()
            if view not in ("CC", "MLO"):
                continue

            record = CSAWImageRecord(
                filename=key,
                patient_id=patient_id,
                exam_year=exam_year,
                laterality=laterality,
                view=view,
            )

            image_data[patient_id][exam_year][f"{laterality}_{view}"] = record

        if unmatched:
            print(f"[dataset] skipped {unmatched} HDF5 keys without CSV metadata")

        return image_data

    # ------------------------------------------------------------------
    # Multiple-prior sample construction
    # ------------------------------------------------------------------

    def _build_longitudinal_samples(self) -> list[CSAWLongitudinalSample]:
        samples = []

        for patient_id, exams_by_year in self.image_data.items():
            years = sorted(exams_by_year)

            for laterality in ("L", "R"):
                cc, mlo = f"{laterality}_CC", f"{laterality}_MLO"

                valid_years = [
                    year for year in years
                    if cc in exams_by_year[year] and mlo in exams_by_year[year]
                ]

                if self.mode == "test":
                    if self.fixed_cohort:
                        if self.n_priors == 0:
                            for current_year in valid_years:
                                samples.append(
                                    CSAWLongitudinalSample(
                                        patient_id, laterality, current_year, ()
                                    )
                                )
                            continue

                        # Define the test cohort using max_priors.
                        if len(valid_years) < self.max_priors + 1:
                            continue

                        for i, current_year in enumerate(valid_years):
                            available = valid_years[:i][::-1]
                            if len(available) < self.max_priors:
                                continue

                            priors = tuple(available[:self.n_priors])
                            samples.append(
                                CSAWLongitudinalSample(
                                    patient_id, laterality, current_year, priors
                                )
                            )
                    else:
                        for i, current_year in enumerate(valid_years):
                            available = valid_years[:i][::-1]
                            if len(available) < self.min_priors:
                                continue

                            priors = tuple(available[:self.max_priors])
                            samples.append(
                                CSAWLongitudinalSample(
                                    patient_id, laterality, current_year, priors
                                )
                            )

                elif self.n_priors is None:
                    # Variable number of priors: newest first.
                    for i, current_year in enumerate(valid_years):
                        available = valid_years[:i][::-1]
                        if len(available) < self.min_priors:
                            continue

                        priors = tuple(available[:self.max_priors])
                        samples.append(
                            CSAWLongitudinalSample(
                                patient_id, laterality, current_year, priors
                            )
                        )

                else:
                    # Fixed number of priors: consecutive sliding windows,
                    # preserving the original chronological order.
                    window_size = self.n_priors + 1
                    for start in range(len(valid_years) - window_size + 1):
                        window = valid_years[start:start + window_size]
                        samples.append(
                            CSAWLongitudinalSample(
                                patient_id,
                                laterality,
                                window[-1],
                                tuple(window[:-1]),
                            )
                        )

        if self.mode == "train":
            random.Random(self.seed).shuffle(samples)

        counts = defaultdict(int)
        for sample in samples:
            counts[len(sample.prior_years)] += 1

        for n in sorted(counts):
            print(f"  {n} prior(s): {counts[n]} samples")

        print(f"Found {len(samples)} longitudinal samples in '{self.mode}'")
        return samples

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        row = self.row_lookup.get(record.filename)
        if row is None:
            raise KeyError(f"No CSV row matched image {record.filename}")
        return row

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range.")

        sample = self.samples[index]
        patient_id = sample.patient_id
        current_year = sample.current_year
        laterality = sample.laterality

        cc_key = f"{laterality}_CC"
        mlo_key = f"{laterality}_MLO"

        current_exam = self.image_data[patient_id][current_year]
        current_cc = current_exam[cc_key]
        current_mlo = current_exam[mlo_key]

        current_image_cc = self._load_image(current_cc.filename)
        current_image_mlo = self._load_image(current_mlo.filename)

        previous_images_cc = []
        previous_images_mlo = []
        previous_image_ids_cc = []
        previous_image_ids_mlo = []
        time_gaps = []

        for prior_year in sample.prior_years:
            prior_exam = self.image_data[patient_id][prior_year]
            prior_cc = prior_exam[cc_key]
            prior_mlo = prior_exam[mlo_key]

            previous_images_cc.append(self._load_image(prior_cc.filename))
            previous_images_mlo.append(self._load_image(prior_mlo.filename))

            previous_image_ids_cc.append(prior_cc.filename)
            previous_image_ids_mlo.append(prior_mlo.filename)

            time_gaps.append(
                min(abs(current_year - prior_year), self.n_years)
            )

        # Apply one consistent flip to current and prior images.
        if self.mode == "train" and torch.rand(()) < 0.5:
            current_image_cc = torch.flip(current_image_cc, dims=[-1])
            current_image_mlo = torch.flip(current_image_mlo, dims=[-1])
            previous_images_cc = [
                torch.flip(image, dims=[-1]) for image in previous_images_cc
            ]
            previous_images_mlo = [
                torch.flip(image, dims=[-1]) for image in previous_images_mlo
            ]

        current_row = self._lookup_row(current_cc)

        ttc = clean_time_to_cancer(current_row["years_to_cancer"])
        followup = clean_followup_years(
            current_row["years_to_last_followup"]
        )

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years, ttc, followup
        )

        return {
            "current_image_cc": current_image_cc,
            "current_image_mlo": current_image_mlo,
            "previous_image_cc": previous_images_cc,
            "previous_image_mlo": previous_images_mlo,
            "current_image_id_cc": current_cc.filename,
            "current_image_id_mlo": current_mlo.filename,
            "previous_image_ids_cc": previous_image_ids_cc,
            "previous_image_ids_mlo": previous_image_ids_mlo,
            "time_gap": torch.tensor(time_gaps, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "y_mask": torch.as_tensor(y_mask, dtype=torch.float32),
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "years_to_cancer": ttc - 1 if ttc is not None else ttc,
            "years_to_last_followup": followup,
            "density": map_density(current_row["density_group"]),
            "patient_id": patient_id,
            "cancer_type": map_cancer_type(current_row["x_type"]),
            "n_priors": len(sample.prior_years),
        }
