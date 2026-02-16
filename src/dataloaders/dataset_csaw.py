
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

class BreastCancerRiskDatasetCSAWCC(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index('file_name').to_dict(orient='index')
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode
        self.image_data = self._load_image_data()
        self.patient_view_pairs = self._create_longitudinal_samples()
        self.num_elements = len(self.patient_view_pairs)

    def map_density(self, value):
        mapping = {
            'A': 1,
            'B': 2,
            'C': 3,
        }
        index = mapping.get(value, -1)  # -1 for NA or unknown values
        return torch.tensor(index, dtype=torch.long)

    def map_cancer_type(self, value):
        """Map cancer type (0,1,2) to integer tensor suitable for GPU ops.
           Return -1 if value is missing or outside {0,1,2}.
        """
        mapping = { 1: 1, 2: 2, 3:3}
        index = mapping.get(value, -1)
        return torch.tensor(index, dtype=torch.long)

    def _load_image_data(self):
        """
        Load image data into a nested dictionary:
        { patient_id -> { exam_year -> { view -> metadata } } }
        """
        image_data = defaultdict(lambda: defaultdict(dict))  # Nested defaultdict
        csv_data = self.csv_data
        image_dir_path = os.path.join(self.data_dir, self.mode).replace("\\", "/")
        for filename in os.listdir(image_dir_path):
            if filename in csv_data:
                file_info = csv_data[filename]
                # Extract required fields
                patient_id = str(file_info['patient_id'])
                exam_year = file_info['exam_year']
                laterality = file_info['laterality']  # Left or Right
                view = file_info['viewposition']  # CC or MLO
                years_to_cancer = file_info['years_to_cancer']
                years_last_followup = file_info['years_to_last_followup']
                density = file_info['density_group']
                cancer_type = file_info['x_type']
                # Map laterality to L and R for consistency
                laterality = 'L' if laterality == 'Left' else 'R' if laterality == 'Right' else None
                # Skip if laterality is invalid
                if laterality is None:
                    print(f"Skipping file due to invalid laterality: {filename}")
                    continue
                # Group by patient_id, exam_year, and view
                key = f"{laterality}_{view}"
                image_data[patient_id][exam_year][key] = {
                    'filename': filename,
                    'years_to_cancer': years_to_cancer,
                    'years_last_followup': years_last_followup,
                    'density': density,
                    'cancer_type':cancer_type,
                }
        return image_data


    def _create_longitudinal_samples(self):
        """
        Pre-compute longitudinal samples per patient.
        """
        samples = []
        for patient_id, year_dict in self.image_data.items():
            if len(year_dict) < 2:
                continue  # Skip patients with fewer than 2 exam years
            # Sort years in ascending order
            sorted_years = sorted(year_dict.keys())
            for idx in range(1, len(sorted_years)):
                prior_year = sorted_years[idx - 1]
                current_year = sorted_years[idx]
                # Check for both CC and MLO views for the current and prior years
                for laterality in ['L', 'R']:
                    cc_view = f"{laterality}_CC"
                    mlo_view = f"{laterality}_MLO"
                    if (cc_view in year_dict[current_year] and mlo_view in year_dict[current_year] and
                            cc_view in year_dict[prior_year] and mlo_view in year_dict[prior_year]):
                        samples.append((patient_id, laterality, current_year, prior_year))
        random.shuffle(samples)
        print(f"Found {len(samples)} valid longitudinal multi-view samples in '{self.mode}' dataset.")
        return samples

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        """
        Fetches a complete longitudinal multi-view sample.
        Returns a dictionary containing the current and prior CC and MLO images.
        """
        # Get the pre-computed sample information
        patient_id, laterality, current_year, prior_year = self.patient_view_pairs[idx]
        # Retrieve all four filenames
        cc_view = f"{laterality}_CC"
        mlo_view = f"{laterality}_MLO"
        current_cc_file = self.image_data[patient_id][current_year][cc_view]['filename']
        current_mlo_file = self.image_data[patient_id][current_year][mlo_view]['filename']
        prior_cc_file = self.image_data[patient_id][prior_year][cc_view]['filename']
        prior_mlo_file = self.image_data[patient_id][prior_year][mlo_view]['filename']

        def load_and_process(filename):
            """Helper to load, convert, normalize, and transform an image."""
            img_path = os.path.join(self.data_dir, self.mode, filename)
            img_pil = Image.open(img_path)  # Ensure grayscale
            img_np = np.array(img_pil)
            img_tensor = torch.from_numpy(img_np).to(torch.float32)
            img_tensor = img_tensor / 256.0
            if self.transform:
                img_tensor = self.transform(img_tensor)
                img_tensor = img_tensor.squeeze(0)
            else:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        # Load and process images
        current_image_cc = load_and_process(current_cc_file)
        current_image_mlo = load_and_process(current_mlo_file)
        previous_image_cc = load_and_process(prior_cc_file)
        previous_image_mlo = load_and_process(prior_mlo_file)

        if self.mode == 'train':
            images = torch.stack([current_image_cc, current_image_mlo, previous_image_cc, previous_image_mlo])
            if torch.rand(1).item() < 0.5:  # 50% probability of flipping
                images = torch.flip(images, dims=[-1])  # Flip horizontally
            current_image_cc, current_image_mlo, previous_image_cc, previous_image_mlo = images

        # Fetch metadata (primarily based on the current screening date)
        current_metadata = self.image_data[patient_id][current_year][cc_view]
        years_last_followup = current_metadata['years_last_followup']
        time_to_cancer = current_metadata['years_to_cancer']
        density = self.map_density(current_metadata['density'])
        cancer_type =  self.map_cancer_type(current_metadata["cancer_type"])

        time_gap = abs(current_year - prior_year)
        if time_gap > 5:  # Limit the time gap to 5 years max
            time_gap = 5
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
        # Return the data
        return {
            'current_image_cc': current_image_cc,
            'current_image_mlo': current_image_mlo,
            'previous_image_cc': previous_image_cc,
            'previous_image_mlo': previous_image_mlo,
            'current_image_id_cc': current_cc_file,
            'current_image_id_mlo': current_mlo_file,
            'previous_image_id_cc': prior_cc_file,
            'previous_image_id_mlo': prior_mlo_file,
            'time_gap': torch.tensor(time_gap, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'y_mask': torch.tensor(y_mask, dtype=torch.float32),
            'event_observed': torch.tensor(event_observed, dtype=torch.float32),
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density,
            'patient_id': patient_id,
            'cancer_type': cancer_type,

        }


