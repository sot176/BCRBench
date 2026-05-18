from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
import kornia.augmentation.container as K_C

from utils import (
    load_tabular_data,
    load_image_tensor,
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
    laterality: str
    view: str
    study_date: pd.Timestamp


class BreastCancerRiskDatasetCSAWCCImgFeatAlign(Dataset):
    """
    Longitudinal CSAW-CC mammography dataset.

    Each sample contains:
        - current image
        - previous image
        - longitudinal survival targets
        - metadata
    """

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        mode: str,
        transforms: Optional[Callable] = None,
        n_years: int = 5,
    ) -> None:

        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years

        self.data = load_tabular_data(csv_file)
        self.csv_lookup = self.data.set_index("file_name").to_dict(orient="index")

        self.image_data = self._load_image_data()
        self.image_pairs = self._build_image_pairs()

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):

        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(f"Index {index} outside dataset length {len(self)}.")

        current, previous = self.image_pairs[index]

        current_image = load_image_tensor(self.data_dir / self.mode / current.filename)
        previous_image = load_image_tensor(self.data_dir / self.mode / previous.filename)

        if self.transform:
            current_image = self.transform(current_image).squeeze(0)
            previous_image = self.transform(previous_image).squeeze(0)

            images = torch.stack([current_image, previous_image])

            if self.mode == "train" and torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[-1])

            current_image, previous_image = images

        current_meta = self.csv_lookup[current.filename]
        previous_meta = self.csv_lookup[previous.filename]

        time_to_cancer = clean_time_to_cancer(current_meta["years_to_cancer"])
        time_to_cancer_prev = clean_time_to_cancer(previous_meta["years_to_cancer"])

        followup = clean_followup_years(current_meta["years_to_last_followup"])
        followup_prev = clean_followup_years(previous_meta["years_to_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            time_to_cancer,
            followup,
        )

        target_prev, y_mask_prev, event_time_prev, _ = build_survival_target(
            self.n_years,
            time_to_cancer_prev,
            followup_prev,
        )

        time_gap = min(
            abs(current.study_date.year - previous.study_date.year),
            self.n_years,
        )

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current.filename,
            "previous_image_id": previous.filename,
            "event_observed": torch.tensor(event_observed, dtype=torch.float32),
            "event_times": torch.tensor(event_time, dtype=torch.float32),
            "previous_event_times": torch.tensor(event_time_prev, dtype=torch.float32),
            "time_gap": torch.tensor(time_gap, dtype=torch.float32),
            "y_mask": torch.tensor(y_mask, dtype=torch.float32),
            "y_mask_prior": torch.tensor(y_mask_prev, dtype=torch.float32),
            "years_to_cancer": torch.tensor(time_to_cancer - 1, dtype=torch.float32),
            "years_to_cancer_prior": torch.tensor(time_to_cancer_prev - 1, dtype=torch.float32),
            "years_to_last_followup": torch.tensor(followup, dtype=torch.float32),
            "years_to_last_followup_prior": torch.tensor(followup_prev, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_prior": torch.tensor(target_prev, dtype=torch.float32),
            "density": map_density(current_meta["density"]),
            "cancer_type": map_cancer_type(current_meta["cancer_type"]),
        }

    def _load_image_data(self):

        image_data = defaultdict(list)
        image_folder = self.data_dir / self.mode

        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        for image_path in image_folder.iterdir():

            filename = image_path.name

            if filename not in self.csv_lookup:
                continue

            meta = self.csv_lookup[filename]

            record = CSAWImageRecord(
                filename=filename,
                patient_id=str(meta["patient_id"]),
                laterality=meta["laterality"],
                view=meta["viewposition"],
                study_date=pd.Timestamp(meta["exam_year"]),
            )

            key = (record.patient_id, record.laterality, record.view)
            image_data[key].append(record)

        for key in image_data:
            image_data[key].sort(key=lambda x: x.study_date)

        return image_data

    def _build_image_pairs(self):

        pairs = []

        for records in self.image_data.values():

            if len(records) < 2:
                continue

            for idx in range(len(records) - 1):
                pairs.append((records[idx], records[idx + 1]))

        random.shuffle(pairs)

        print(
            f"Found {len(pairs)} valid longitudinal image pairs "
            f"in '{self.mode}' dataset."
        )

        return pairs
    




def main():
    path_data_train_val = "C:/UiT_PhD_datasets/csaw_dataset/Risk_dataset_train_val_test_1664_2048_new"
    csv_file = "C:/Users/sothr1456/OneDrive - UiT Office 365/Documents/UiT PhD/CSAW/metadata_csawcc_dcm_path_density_new.csv"
    train_transform =torch.nn.Sequential(
        K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.05), degrees=0, shear=0, p=0.5),
        K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
        K_A.RandomGamma(gamma=(0.8, 1.4), gain=(0.95, 1.05), p=0.5),
        K_A.RandomCrop(size=(1843, 1498), p=0.2),
        K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
    )

    train_dataset = BreastCancerRiskDatasetCSAWCCImgFeatAlign(csv_file, path_data_train_val, 'train',
                                             transforms=train_transform)
    train_loader = DataLoader( train_dataset, batch_size=10,
                              shuffle=True, pin_memory=True)


    batch = next(iter(train_loader))  # grab a batch from the dataloader
    img_current = batch['current_image']
    img_previous = batch['previous_image']
    event_times = batch['event_times']
    event_observed = batch['event_observed']
    target_prior = batch['target_prior']
    time_gap = batch['time_gap']
    target = batch['target']
    density = batch['density']
    cancer_type = batch['cancer_type']

    print('cancer_type',cancer_type)
    y_mask = batch['y_mask']
    y_mask_pri = batch['y_mask_prior']
    print(event_observed)
    print(event_times)
    print(density)

    print("time gap", time_gap)
    print("mask", y_mask)
    print("target", target)
    print("mask prior", y_mask_pri)
    print("target_prior", target_prior)
    for i in range(8):
        print(batch['current_image_id'][i], batch['previous_image_id'][i])
        print(img_current.shape)
        current_image_float32 = img_current[i].squeeze().cpu().numpy()  # Convert to float32 and move to CPU
        previous_image_float32 = img_previous[i].squeeze().cpu().numpy()  # Same for previous image
        plt.subplot(1, 2, 1)
        plt.imshow(img_current[i].squeeze(), cmap='gray')
        plt.title('Current Image')
        # Plot the previous image in the second subplot (1 row, 2 columns, 2nd position)
        plt.subplot(1, 2, 2)
        plt.imshow(img_previous[i].squeeze(), cmap='gray')
        plt.title('Previous Image')
        plt.show()
    label = batch['label']
    print(label)

if __name__ == '__main__':
    main()