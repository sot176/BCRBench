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


def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)

class BreastCancerRiskDatasetCSAW_Mirai(Dataset):
    """
    One sample = one exam (4 images only)
    No longitudinal pairing.
    """

    def __init__(
        self,
        csv_file,
        image_dir,
        mode,
        transforms: Optional[Callable] = None,
        n_years: int = 5,
    ):
        self.image_dir = image_dir
        self.mode = mode
        self.transforms = transforms
        self.n_years = n_years

        self.df = pd.read_csv(csv_file, low_memory=False)

        self.records = self._build_records()
        self.exams = self._group_into_exams()

    def _build_records(self):
        records = []

        for _, row in self.df.iterrows():

            laterality = "L" if row["laterality"] == "Left" else "R"

            records.append(
                CSAWImageRecord(
                    filename=row["file_name"],
                    patient_id=str(row["patient_id"]),
                    laterality=laterality,
                    view=row["viewposition"],
                    study_date=pd.Timestamp(row["exam_year"]),
                )
            )

        return records

    def _group_into_exams(self):
        exams = {}

        for r in self.records:
            key = (r.patient_id, r.study_date)

            exams.setdefault(key, {}).setdefault(
                f"{r.laterality}_{r.view}", r
            )

        required = {"L_CC", "L_MLO", "R_CC", "R_MLO"}

        exams = {
            k: v for k, v in exams.items()
            if required.issubset(v.keys())
        }

        self.exam_list = list(exams.items())

        print(f"[INFO] Found {len(self.exam_list)} complete 4-view exams")

        return exams

    def __len__(self):
        return len(self.exam_list)

def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | np.ndarray]:

    exam_id, views = self.exam_list[index]

    order = ["L_CC", "L_MLO", "R_CC", "R_MLO"]

    # -------------------------
    # load images
    # -------------------------
    def load(record: CSAWImageRecord):
        path = os.path.join(self.image_dir, self.mode, record.filename)

        img = np.array(Image.open(path)).astype(np.float32) / 65535.0
        t = torch.from_numpy(img)

        if self.transforms:
            t = self.transforms(t).squeeze(0)
        else:
            t = t.unsqueeze(0)

        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)

        return t

    images = torch.stack([load(views[v]) for v in order])  # [4,3,H,W]

    # -------------------------
    # metadata source (any view works, pick CC)
    # -------------------------
    meta = views["L_CC"]

    patient_id = meta.patient_id

    # -------------------------
    # survival labels
    # -------------------------
    ttc = self._clean(meta.years_to_cancer)
    follow = self._clean(meta.years_to_last_followup)

    target, y_mask, event_time, event_observed = self._build_survival(ttc, follow)

    # -------------------------
    # sequences (MIRAI-style)
    # NOTE: single exam → static tokens
    # -------------------------
    all_views = [0, 1, 0, 1]
    all_sides = [1, 1, 0, 0]
    all_times = [0, 0, 0, 0]

    time_seq = pad_to_length(all_times, MAX_TIME, len(all_times))
    view_seq = pad_to_length(all_views, MAX_VIEWS, len(all_views))
    side_seq = pad_to_length(all_sides, MAX_SIDES, len(all_sides))

    # -------------------------
    # RETURN (your EMBED-style format)
    # -------------------------
    return {
        "images": images,  # [4, 3, H, W]

        "exam_id": exam_id,
        "patient_id": patient_id,

        "target": torch.tensor(target, dtype=torch.float32),
        "y_mask": torch.tensor(y_mask, dtype=torch.float32),
        "event_observed": torch.tensor(event_observed, dtype=torch.float32),
        "event_times": torch.tensor(event_time, dtype=torch.float32),

        "density": self._map_density(meta.density_group),
        "cancer_type": self._map_cancer_type(meta.x_type),

        "time_seq": torch.tensor(time_seq, dtype=torch.long),
        "view_seq": torch.tensor(view_seq, dtype=torch.long),
        "side_seq": torch.tensor(side_seq, dtype=torch.long),
    }
    





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