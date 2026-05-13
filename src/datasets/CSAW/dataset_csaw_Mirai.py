import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset 
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable
 

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
import kornia.augmentation.container as K_C

MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=20


@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp

@dataclass(frozen=True)
class CSAWExamSample:
    patient_id: str
    exam_year: int
    views: dict[str, CSAWImageRecord]

def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)

class BreastCancerRiskDatasetCSAW_Mirai(Dataset):

    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index("file_name").to_dict(orient="index")

        self.data_dir = image_dir
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years

        self.image_data = self._load_image_data()
        self.samples  = self._build_samples()
 

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        left_cc = sample.views["L_CC"]
        left_mlo = sample.views["L_MLO"]

        right_cc = sample.views["R_CC"]
        right_mlo = sample.views["R_MLO"]

        # -------------------------
        # Load images
        # -------------------------
        left_cc_image = self._load_image(
            left_cc.filename
        )

        left_mlo_image = self._load_image(
            left_mlo.filename
        )

        right_cc_image = self._load_image(
            right_cc.filename
        )

        right_mlo_image = self._load_image(
            right_mlo.filename
        )

        # -------------------------
        # Transforms
        # -------------------------
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

        # -------------------------
        # Stack views
        # [4, C, H, W]
        # -------------------------
        images = torch.stack(
            [
                left_cc_image,
                left_mlo_image,
                right_cc_image,
                right_mlo_image,
            ]
        )

        # -------------------------
        # Metadata
        # use one canonical row
        # -------------------------
        current_row = self.csv_data[
            right_cc.filename
        ]

        current_time_to_cancer = self._clean_time(
            current_row["years_to_cancer"]
        )

        years_last_followup = self._clean_followup(
            current_row["years_to_last_followup"]
        )

        target, y_mask, event_time, event_observed = (
            self._build_survival_target(
                current_time_to_cancer,
                years_last_followup,
            )
        )

        # -------------------------
        # MIRAI temporal/view tokens
        # -------------------------
        all_views = [0, 1, 0, 1]
        all_sides = [0, 0, 1, 1]
        all_times = [0, 0, 0, 0]

        time_seq = pad_to_length(all_times, MAX_TIME, len(all_times))
        view_seq = pad_to_length(all_views, MAX_VIEWS, len(all_views))
        side_seq = pad_to_length(all_sides, MAX_SIDES, len(all_sides))

        return {
            "images": images,
            "target": torch.tensor(target, dtype=torch.float32,),
            "y_mask": torch.tensor(y_mask, dtype=torch.float32,),
            "event_observed": torch.tensor(event_observed,dtype=torch.float32,),
            "event_times": torch.tensor(event_time, dtype=torch.float32,),
            "density": self._map_density(current_row["density_group"]),
            "cancer_type": self._map_cancer_type(current_row["x_type"]),
            "patient_id": sample.patient_id,
            "time_seq": torch.tensor(time_seq,dtype=torch.long, ),
            "view_seq": torch.tensor(view_seq,dtype=torch.long,),
            "side_seq": torch.tensor(side_seq,dtype=torch.long,),
            "years_to_cancer": current_time_to_cancer - 1,
            "years_to_last_followup": years_last_followup,
        }

    def _load_image_data(self):
        image_data = defaultdict(list)
        base_dir = os.path.join(self.data_dir, self.mode)

        for filename in os.listdir(base_dir):

            if filename not in self.csv_data:
                continue

            meta = self.csv_data[filename]

            record = CSAWImageRecord(
                filename=filename,
                patient_id=str(meta["patient_id"]),
                laterality=meta["laterality"],
                view=meta["viewposition"],
                study_date=pd.Timestamp(meta["exam_year"]),
            )

            key = (
                record.patient_id,
                record.laterality,
                record.view,
            )

            image_data[key].append(record)

        return image_data

    def _build_samples(self):

        exams = defaultdict(dict)

        # regroup by patient + study_date
        for records in self.image_data.values():

            for record in records:

                exam_key = (
                    record.patient_id,
                    record.study_date,
                )

                view_key = (
                    f"{record.laterality}_{record.view}"
                )

                exams[exam_key][view_key] = record

        required_views = {
            "L_CC",
            "L_MLO",
            "R_CC",
            "R_MLO",
        }

        samples = []

        for (patient_id, study_date), views in exams.items():

            if required_views.issubset(views.keys()):

                samples.append(
                    CSAWExamSample(
                        patient_id=patient_id,
                        exam_year=study_date,
                        views=views,
                    )
                )

        print(f"[INFO] Found {len(samples)} complete exams")

        return samples
    
    def _load_image(self, filename):
        path = os.path.join(self.data_dir, self.mode, filename)

        with Image.open(path) as img:
            img = np.array(img).astype(np.float32)

        img = self._normalize_uint16(img)

        tensor = torch.from_numpy(img).float()
        return tensor.unsqueeze(0)

    @staticmethod
    def _normalize_uint16(img: np.ndarray):
        img = img.astype(np.float32)
        return img / 65535.0

    def _build_survival_target(self, ttc, followup):
        target = np.zeros(self.n_years, dtype=np.int8)

        event_time = ttc - 1
        event_observed = int(event_time < self.n_years)

        if event_observed:
            target[event_time:] = 1
        else:
            event_time = min(followup, self.n_years) - 1

        mask = np.zeros(self.n_years, dtype=np.int8)
        mask[: event_time + 1] = 1

        return target, mask, event_time, event_observed

    @staticmethod
    def _clean_time(x):
        if pd.isna(x):
            return 6
        x = int(x)
        return 1 if x == 0 else x

    @staticmethod
    def _clean_followup(x):
        if pd.isna(x):
            return 1
        x = int(x)
        return 1 if x == 0 else x

    @staticmethod
    def _map_density(x):
        return torch.tensor({"A": 1, "B": 2, "C": 3}.get(x, -1))

    @staticmethod
    def _map_cancer_type(x):
        return torch.tensor({1: 1, 2: 2, 3: 3}.get(x, -1))




def main():
    path_data_train_val = "C:/UiT_PhD_datasets/csaw_dataset/Risk_dataset_train_val_test_1664_2048_new"
    csv_file = "C:/Users/sothr1456/OneDrive - UiT Office 365/Documents/UiT PhD/CSAW/metadata_csawcc_dcm_path_density_new.csv"
    #path_data_train_val = "D:/UiT_PhD_datasets/csaw_dataset/Risk_dataset_train_val_test_1664_2048_new"
    #csv_file = "D:/UiT PhD/CSAW/metadata_csawcc_dcm_path_new.csv"

    train_transform = K_C.AugmentationSequential(
        K_A.RandomCrop(size=(1946, 1581), p=0.2),
        K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
        K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.1), degrees=0, shear=0, p=0.5),
        K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
        K_A.RandomGamma(gamma=(0.8, 1.2), gain=(0.9, 1.05), p=0.5),
    )
    train_dataset = BreastCancerRiskDatasetCSAW_Mirai(csv_file, path_data_train_val, 'test',
                                             transforms=train_transform)
    if len(train_dataset) == 0:
        print("Dataset is empty. Please check paths and data processing logic.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Important for Windows debugging
    )

    # --- 3. Fetch and Inspect a Single Batch ---
    print("\nFetching a batch from the DataLoader...")
    batch = next(iter(train_loader))

    images = batch['images']  # [B, 4, H, W]
    print("images shape", images.shape)
    event_observed = batch['event_observed']
    event_times = batch['event_times']
    target = batch['target']
    y_mask = batch['y_mask']
    cancer_type = batch['cancer_type']
    density = batch['density']
    patient_id = batch['patient_id']
    time_seq = batch['time_seq']
    view_seq = batch['view_seq']
    side_seq = batch['side_seq']
    print("time seq", time_seq)
    print("view seq", view_seq)
    print("side seq", side_seq)
    # --- Print metadata for each sample ---
    for i in range(len(images)):
        print(f"\nSample {i + 1} | Patient ID: {patient_id[i]}")

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(10, 12))
        fig.suptitle(f"Patient {patient_id[i]}", fontsize=16)

        view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
        for v_idx, view_name in enumerate(view_names):
            img = images[i, :, v_idx, :, :]  # shape [C, H, W]

            # Convert to 2D array for grayscale plotting
            if img.shape[0] == 3:
                img_to_show = img[0].cpu().numpy()  # use one channel
            else:
                img_to_show = img.squeeze(0).cpu().numpy()  # for single-channel

            row = 0 if v_idx < 2 else 1
            col = v_idx % 2
            axes[row, col].imshow(img_to_show, cmap='gray')
            axes[row, col].set_title(view_name, fontsize=14)
            axes[row, col].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    main()