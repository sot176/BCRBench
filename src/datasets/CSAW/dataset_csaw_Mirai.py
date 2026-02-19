
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
import kornia.augmentation.container as K_C

MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=20

def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)


def imgunit16(img):
    # mammogram_dicom = img
    # orig_min = mammogram_dicom.min()
    # orig_max = mammogram_dicom.max()
    # target_min = 0.0
    # target_max = 255.0
    # mammogram_scaled = (mammogram_dicom-orig_min)*((target_max-target_min)/(orig_max-orig_min))+target_min
    mammogram_scaled = (img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 65535

    # print(mammogram_uint8_by_function.max(), mammogram_uint8_by_function.min())
    # mammogram_uint8_by_function = mammogram_uint8_by_function
    return mammogram_scaled

class BreastCancerRiskDatasetCSAWCC_Mirai(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index('file_name').to_dict(orient='index')
        self.image_dir = image_dir
        self.transforms = transforms
        self.n_years = n_years
        self.mode = mode
        self.image_data = self._load_metadata()
        self.samples = self._create_samples()

    def map_density(self, value):
        mapping = {'A': 1, 'B': 2, 'C': 3}
        return torch.tensor(mapping.get(value, -1), dtype=torch.long)

    def map_cancer_type(self, value):
        mapping = {1: 1, 2: 2, 3: 3}
        return torch.tensor(mapping.get(value, -1), dtype=torch.long)

    def _load_metadata(self):
        """
        Load only metadata (no images) into nested dict:
        {patient_id -> exam_year -> {view -> metadata}}
        """
        image_data = defaultdict(lambda: defaultdict(dict))
        image_dir_path = os.path.join(self.image_dir, self.mode)

        for filename in os.listdir(image_dir_path):
            if filename not in self.csv_data:
                continue

            meta = self.csv_data[filename]
            patient_id = str(meta['patient_id'])
            exam_year = meta['exam_year']
            laterality = 'L' if meta['laterality'] == 'Left' else 'R' if meta['laterality'] == 'Right' else None
            view = meta['viewposition']

            if laterality is None:
                continue

            key = f"{laterality}_{view}"
            meta_copy = meta.copy()
            meta_copy['filename'] = filename
            image_data[patient_id][exam_year][key] = meta_copy

        return image_data

    def _create_samples(self):
        samples = []
        required_views = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
        for patient_id, year_dict in self.image_data.items():
            for year, views in year_dict.items():
                if all(v in views for v in required_views):
                    samples.append((patient_id, year, views))
        print(f"[INFO] Found {len(samples)} exams with 4 views for mode '{self.mode}'.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, year, views = self.samples[idx]

        def load_image(filename):
            img_path = os.path.join(self.image_dir, self.mode, filename)
            img = Image.open(img_path)
            img = np.array(img, dtype=np.float32) / 65535.0
            tensor = torch.from_numpy(img)
            if self.transforms:
                tensor = self.transforms(tensor).squeeze(0)
            else:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            return tensor

        # Load 4 images lazily
        l_cc = load_image(views['L_CC']['filename'])
        l_mlo = load_image(views['L_MLO']['filename'])
        r_cc = load_image(views['R_CC']['filename'])
        r_mlo = load_image(views['R_MLO']['filename'])
        images = torch.stack([l_cc, l_mlo, r_cc, r_mlo])

        # Metadata
        meta = views['L_CC']
        years_last_followup = int(meta['years_to_last_followup'])
        time_to_cancer = int(meta['years_to_cancer'])
        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):  # Check if the value is NaN
            time_to_cancer = 6

        time_to_cancer = time_to_cancer - 1
        any_cancer = time_to_cancer < 5

        target = torch.zeros(5, dtype=torch.float32)
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            target[time_at_event:] = 1
        else:
            # If no cancer, use the last follow-up year
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, 5) - 1)
            event_observed = 0
        y_mask = np.array([1] * (time_at_event + 1) + [0] * (5 - (time_at_event + 1)), dtype=np.int8)
        y_mask = y_mask[:5]

        density = self.map_density(meta['density_group'])
        cancer_type = self.map_cancer_type(meta['x_type'])

        all_views = [0, 1, 0, 1]
        all_sides = [0, 0, 1, 1]
        time_stamps = [0]*len(images)

        time_seq = pad_to_length(time_stamps, MAX_TIME, len(images))
        view_seq = pad_to_length(all_views, MAX_VIEWS, len(images))
        side_seq = pad_to_length(all_sides, MAX_SIDES, len(images))
        images = images.permute(1, 0, 2, 3)  # [4, 3, H, W] → [3, 4, H, W]

        return {
            'images': images,
            'target': target,
            'y_mask': y_mask,
            'event_observed': event_observed,
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density,
            'cancer_type': cancer_type,
            'patient_id': patient_id,
            'time_seq': torch.tensor(time_seq, dtype=torch.long),
            'view_seq': torch.tensor(view_seq, dtype=torch.long),
            'side_seq': torch.tensor(side_seq, dtype=torch.long),
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