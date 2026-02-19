
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

class BreastCancerRiskDatasetCSAWCC_VMRA(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index('file_name').to_dict(orient='index')
        self.image_dir = image_dir
        self.transforms = transforms
        self.n_years = n_years
        self.mode = mode
        self.image_data = self._load_metadata()
        self.samples = self._create_samples_with_priors()

    # ------------------ MAPPERS ------------------

    def map_density(self, value):
        mapping = {'A': 1, 'B': 2, 'C': 3}
        return torch.tensor(mapping.get(value, -1), dtype=torch.long)

    def map_cancer_type(self, value):
        mapping = {1: 1, 2: 2, 3: 3}
        return torch.tensor(mapping.get(value, -1), dtype=torch.long)

    # ------------------ METADATA LOADING ------------------

    def _load_metadata(self):
        """
        Load metadata into nested dict:
        {patient_id -> exam_year -> {view -> metadata_dict}}
        """
        image_data = defaultdict(lambda: defaultdict(dict))
        image_dir_path = os.path.join(self.image_dir, self.mode)

        for filename in os.listdir(image_dir_path):
            if filename not in self.csv_data:
                continue

            meta = self.csv_data[filename]
            patient_id = str(meta['patient_id'])
            exam_year = meta['exam_year']

            laterality = (
                'L' if meta['laterality'] == 'Left' else
                'R' if meta['laterality'] == 'Right' else None
            )
            view = meta['viewposition']
            if laterality is None:
                continue

            key = f"{laterality}_{view}"
            meta_copy = meta.copy()
            meta_copy['filename'] = filename
            image_data[patient_id][exam_year][key] = meta_copy

        return image_data

    # ------------------ SAMPLE CREATION ------------------

    def _create_samples_with_priors(self):
        """
        Only include samples that have BOTH current and prior exams (with all 4 views).
        """
        samples = []
        required_views = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']

        for patient_id, year_dict in self.image_data.items():
            sorted_years = sorted(year_dict.keys())

            # Start from 1 because we need a prior (previous exam)
            for i in range(1, len(sorted_years)):
                curr_year = sorted_years[i]
                prior_year = sorted_years[i - 1]
                curr_views = year_dict[curr_year]
                prior_views = year_dict[prior_year]

                if all(v in curr_views for v in required_views) and all(v in prior_views for v in required_views):
                    samples.append((patient_id, curr_year, curr_views, prior_views))

        print(f"[INFO] Found {len(samples)} samples with prior exams for mode '{self.mode}'.")
        return samples

    # ------------------ LENGTH ------------------

    def __len__(self):
        return len(self.samples)

    # ------------------ ITEM ACCESS ------------------

    def __getitem__(self, idx):
        patient_id, curr_year, curr_views, prior_views = self.samples[idx]

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

        # Load both prior and current 4 views
        def load_views(view_dict):
            return torch.stack([
                load_image(view_dict['L_CC']['filename']),
                load_image(view_dict['L_MLO']['filename']),
                load_image(view_dict['R_CC']['filename']),
                load_image(view_dict['R_MLO']['filename']),
            ])

        curr_imgs = load_views(curr_views)
        prior_imgs = load_views(prior_views)
        all_imgs = torch.stack([prior_imgs, curr_imgs], dim=0)  # [2, 4, 3, H, W]
        all_imgs = all_imgs.permute(0, 2,1, 3, 4)  #  [2, 4, 3, H, W]→ [2, 3, 4, H, W]

        # Use metadata from current exam
        meta = curr_views['L_CC']
        years_last_followup = int(meta['years_to_last_followup'])
        time_to_cancer = meta['years_to_cancer']
        if pd.isna(time_to_cancer):
            time_to_cancer = 6
        if time_to_cancer == 0:
            time_to_cancer = 1

        time_to_cancer = int(time_to_cancer) - 1
        any_cancer = time_to_cancer < 5

        target = torch.zeros(5, dtype=torch.float32)
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            target[time_at_event:] = 1
        else:
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, 5) - 1)
            event_observed = 0

        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * (5 - (time_at_event + 1)),
            dtype=np.int8
        )[:5]

        density = self.map_density(meta['density_group'])
        cancer_type = self.map_cancer_type(meta['x_type'])

        all_images = ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']
        all_views = [0, 1, 0, 1]
        all_sides = [0, 0, 1, 1]
        time_stamps = [0] * len(all_images)

        time_seq = pad_to_length(time_stamps, MAX_TIME, len(all_images))
        view_seq = pad_to_length(all_views, MAX_VIEWS, len(all_images))
        side_seq = pad_to_length(all_sides, MAX_SIDES, len(all_images))

        return {
            'images': all_imgs,
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
    train_dataset = BreastCancerRiskDatasetCSAW_VMRA(csv_file, path_data_train_val, 'test',
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
        pid = patient_id[i]
        print(f"\nSample {i + 1} | Patient ID: {pid}")

        # Create a 2x4 grid: top row = PRIOR, bottom row = CURRENT
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Patient {pid} - Prior (top) vs Current (bottom)", fontsize=16)

        view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']

        for t_idx, time_label in enumerate(['Prior', 'Current']):  # 0=prior, 1=current
            for v_idx, view_name in enumerate(view_names):
                img = images[i, t_idx, v_idx]  # [3, H, W]
                img_to_show = img[0].cpu().numpy() if img.shape[0] == 3 else img.squeeze(0).cpu().numpy()

                axes[t_idx, v_idx].imshow(img_to_show, cmap='gray')
                axes[t_idx, v_idx].set_title(f"{time_label} - {view_name}", fontsize=12)
                axes[t_idx, v_idx].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == '__main__':
    main()