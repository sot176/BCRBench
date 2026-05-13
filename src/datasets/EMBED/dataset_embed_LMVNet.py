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
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample

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


# Backwards compatibility
imgunit16 = scale_to_uint16_range


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

        target, y_mask, event_time, event_observed = (
            self._build_survival_target(
                current_time_to_cancer,
                years_last_followup,
            )
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
            "event_observed": torch.tensor(
                event_observed,
                dtype=torch.float32,
            ),
            "event_times": torch.tensor(
                event_time,
                dtype=torch.float32,
            ),
            "years_to_cancer": current_time_to_cancer - 1,
            "years_to_last_followup": years_last_followup,
            "density": self._map_density(current_row["density"]),
            "cancer_type": self._map_cancer_type(
                current_row["path_severity"]
            ),
            "race": self._map_race(
                current_row.get("RACE_DESC")
            ),
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
    


def main():
    path_data_train_val = "C:/UiT_PhD_datasets/embed_dataset/risk_dataset_1664_2048/"

    csv_file = "C:/Users/sothr1456/OneDrive - UiT Office 365/Documents/UiT PhD/EMBED/combined_cases_with_follow_up_races_new.csv"

    import kornia.augmentation.container as K_C

    train_transform = K_C.AugmentationSequential(
        K_A.RandomCrop(size=(1946, 1581), p=0.2),
        K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
        K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.1), degrees=0, shear=0, p=0.5),
        K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
        K_A.RandomGamma(gamma=(0.8, 1.2), gain=(0.9, 1.05), p=0.5),
    )

    BATCH_SIZE = 12

    print("Initializing dataset...")
    train_dataset = BreastCancerRiskDatasetEMBEDLMVNet(
        csv_file=csv_file,
        image_dir=path_data_train_val,
        mode='train',
        transforms=train_transform
    )

    if len(train_dataset) == 0:
        print("Dataset is empty. Please check paths and data processing logic.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Important for Windows debugging
    )

    # --- 3. Fetch and Inspect a Single Batch ---
    print("\nFetching a batch from the DataLoader...")
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        print("DataLoader is empty. Could not fetch a batch.")
        return

    # Extract the relevant tensors and data from the batch dictionary
    current_image_cc = batch['current_image_cc']
    current_image_mlo = batch['current_image_mlo']
    previous_image_cc = batch['previous_image_cc']
    previous_image_mlo = batch['previous_image_mlo']
    patient_id = batch['patient_id']
    print(batch['laterality'])
    print(previous_image_cc.shape)
    time_gap = batch['time_gap']  # time_gap is the same for both views
    y_mask = batch['y_mask']
    event_times = batch['event_times']
    event_observed = batch['event_observed']
    density = batch['density']
    target = batch['target']

    # --- 4. Print Metadata for the Batch ---
    print("\n--- Batch Metadata ---")
    print(f"Batch size: {current_image_cc.shape[0]}")
    print(f"Time Gap (years): {time_gap.numpy()}")
    print(f"Event Observed (1=Cancer, 0=Censored): {event_observed.numpy()}")
    print(f"Time to Event/Censoring (years): {event_times.numpy()}")
    print(f"Breast Density: {density}")
    print(f"Target Vector (risk over 5 years): \n{target.numpy()}")
    print(f"Mask Vector (valid time points): \n{y_mask.numpy()}")
    print("----------------------\n")

    # --- 5. Visualize the Images for Each Sample in the Batch ---
    print("Plotting the four-image quartet for each sample in the batch...")
    num_samples_to_plot = 4

    for i in range(num_samples_to_plot):
        # Create a 2x2 subplot for each patient sample
        fig, axes = plt.subplots(2, 2, figsize=(10, 12))
        fig.suptitle(
            f"Sample {i + 1} from Batch | Time Gap: {time_gap[i].item():.1f} years | Density: {density[i]} Patient_ID:{patient_id[i]}",
            fontsize=16)

        # It's safer to move tensors to CPU for plotting
        img_cur_cc_plot = current_image_cc[i].squeeze().cpu().numpy()
        img_cur_mlo_plot = current_image_mlo[i].squeeze().cpu().numpy()
        img_prev_cc_plot = previous_image_cc[i].squeeze().cpu().numpy()
        img_prev_mlo_plot = previous_image_mlo[i].squeeze().cpu().numpy()

        # Plot Prior CC
        axes[0, 0].imshow(img_prev_cc_plot, cmap='gray')
        axes[0, 0].set_title('Prior CC')
        axes[0, 0].axis('off')

        # Plot Current CC
        axes[0, 1].imshow(img_cur_cc_plot, cmap='gray')
        axes[0, 1].set_title('Current CC')
        axes[0, 1].axis('off')

        # Plot Prior MLO
        axes[1, 0].imshow(img_prev_mlo_plot, cmap='gray')
        axes[1, 0].set_title('Prior MLO')
        axes[1, 0].axis('off')

        # Plot Current MLO
        axes[1, 1].imshow(img_cur_mlo_plot, cmap='gray')
        axes[1, 1].set_title('Current MLO')
        axes[1, 1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.show()


if __name__ == '__main__':
    main()