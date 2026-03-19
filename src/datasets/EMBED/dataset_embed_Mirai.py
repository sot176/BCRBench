import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import re
from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2

def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)

def imgunit16(img):
    mammogram_scaled = (
            (img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 65535
    )
    return mammogram_scaled


def extract_date_from_filename(filename):
    """Extracts a datetime object from a standard filename."""
    # e.g., "12345_R_MLO_2019-07-13_....png" -> "2019-07-13"
    try:
        date_str = filename.split("_")[3]
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (IndexError, ValueError):
        return None  # Return None if filename format is unexpected

MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=10

class BreastCancerRiskDatasetEMBED_Mirai(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        """
        Args:
            csv_file (str): Path to CSV file containing patient data.
            image_dir (str): Directory containing mammogram images.
            transforms (callable, optional): Transform to be applied to images.
            n_years (int): Number of years for risk prediction (default: 5).
        """
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode

        # Ensure date columns are in datetime format
        self.data["study_date_anon"] = pd.to_datetime(
            self.data["study_date_anon"], errors='coerce'
        )

        self.image_data = self._load_image_data()
        self.samples = self._create_samples()

    def map_density(self, value):
        """Map density value to integer tensor."""
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}
        return torch.tensor(mapping.get(value, -1), dtype=torch.long)

    def map_cancer_type(self, value):
        """Map cancer type value to integer tensor."""
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        return torch.tensor(mapping.get(value, -1), dtype=torch.long)

    def _load_image_data(self):
        """Load image data into nested dict: {patient_id -> {date -> {view -> filename}}}."""
        image_data = defaultdict(lambda: defaultdict(dict))
        pattern = r"(\d+)_([RL])_([A-Z]+)_(\d{4}-\d{2}-\d{2})_.*\.png"

        for filename in os.listdir(os.path.join(self.data_dir, self.mode).replace("\\", "/")):
            match = re.match(pattern, filename)
            if match:
                patient_id = match.group(1)
                laterality = match.group(2)
                view_type = match.group(3)
                date_obj = datetime.strptime(match.group(4), "%Y-%m-%d")
                full_view = f"{laterality}_{view_type}"
                image_data[patient_id][date_obj][full_view] = filename

        return image_data

    def _create_samples(self):
        """
        Create samples for each patient on each date with all four views.
        Returns a list of tuples: (patient_id, date, views_dict)
        """
        samples = []
        for patient_id, date_dict in self.image_data.items():
            for date, views in date_dict.items():
                if all(v in views for v in ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']):
                    samples.append((patient_id, date, views))
        print(f"Found {len(samples)} valid samples in '{self.mode}' dataset.")
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_metadata_for_image(self, filename):
        """Helper to get metadata row from CSV for a given image."""
        parts = filename.split("_")
        patient_id, laterality, view = parts[0], parts[1], parts[2]
        study_date = extract_date_from_filename(filename)

        matching_row = self.data[
            (self.data["patient_id"].astype(str) == patient_id) &
            (self.data["ImageLateralityFinal"] == laterality) &
            (self.data["view"] == view) &
            (self.data["study_date_anon"] == study_date)
        ]

        return matching_row.iloc[0] if not matching_row.empty else None

    def __getitem__(self, idx):
        """Return a sample with stacked 4 views and associated metadata."""
        patient_id, date, views = self.samples[idx]

        # --- 1. Load all four images ---
        def load_and_process(filename):
            img_path = os.path.join(self.data_dir, self.mode, filename)
            img = Image.open(img_path)
            img = np.array(img)
            tensor = torch.from_numpy(img) / 65535.0
            if self.transform:
                tensor = self.transform(tensor).squeeze(0)
            else:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0]==1:
                tensor = tensor.repeat(3, 1, 1)
            return tensor

        l_cc = load_and_process(views['L_CC'])
        l_mlo = load_and_process(views['L_MLO'])
        r_cc = load_and_process(views['R_CC'])
        r_mlo = load_and_process(views['R_MLO'])
        images = torch.stack([l_cc, l_mlo, r_cc, r_mlo])  # [4,3, H, W]
        # --- 2. Fetch metadata ---
        metadata = self._get_metadata_for_image(views['R_CC'])
        if metadata is None:
            print(f"Warning: Missing metadata for {patient_id} on {date}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self))

        # --- 3. Process target and mask ---
        years_last_followup = int(metadata["years_last_followup"])
        time_to_cancer = metadata["Time_to_Cancer_Years"]

        # --- 4. Other metadata ---
        density = self.map_density(metadata["density"])
        cancer_type = self.map_cancer_type(metadata["path_severity"])

        # --- 5. Return dictionary ---
        all_images = ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']
        all_views = [0, 1, 0, 1]  # CC=0, MLO=1
        all_sides = [0, 0, 1, 1]  # Right=0, Left=1
        time_stamps = [0] * len(all_images)

       # Fix padding values to match MAX constants:
        time_seq = pad_to_length(time_stamps, MAX_TIME,   len(all_images))  # pad with 10
        view_seq = pad_to_length(all_views,   MAX_VIEWS,  len(all_images))  # pad with 2
        side_seq = pad_to_length(all_sides,   MAX_SIDES,  len(all_images))  # pad with 2
        images = images.permute(1, 0, 2, 3)  # [4, 3, H, W] → [3, 4, H, W]

        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):  # Check if the value is NaN
            time_to_cancer = 6

        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)

        time_to_cancer = time_to_cancer - 1
        any_cancer = time_to_cancer < 5

        # Initialize the sequence with zeros
        target = np.zeros(5, dtype=np.int8)

        # If the patient has cancer, mark the event year and later with 1
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
        # Create a mask for valid time points (before the last event or follow-up)

        data = {
            'images': images,  #  [3, 4, H, W]
            'target': target,  # [5]
            'y_mask': y_mask,  # [5]
            'event_observed': event_observed,
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density,
            'patient_id': patient_id,
            'cancer_type': cancer_type,
            'time_seq': torch.tensor(time_seq, dtype=torch.long),
            'view_seq': torch.tensor(view_seq, dtype=torch.long),
            'side_seq': torch.tensor(side_seq, dtype=torch.long),
        }
        return data



def main():
    path_data_train_val = "C:/UiT_PhD_datasets/embed_dataset/risk_dataset_1664_2048/"

    csv_file = "C:/Users/sothr1456/OneDrive - UiT Office 365/Documents/UiT PhD/EMBED/combined_cases_with_followup.csv"

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
    train_dataset = BreastCancerRiskDataset_Mirai(
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
    print(" y mask", y_mask)
    print("envent times", event_times)
    print("event_observed", event_observed)
    patient_id = batch['patient_id']
    time_seq = batch['time_seq']
    view_seq = batch['view_seq']
    side_seq = batch['side_seq']
    print("time seq shape", time_seq.shape)
    print("time seq", time_seq)
    print("view seq", view_seq)
    print("side seq", side_seq)
    # --- Print metadata for each sample ---
    for i in range(1):
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