import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import re
from collections import defaultdict
import random
from datetime import datetime
import numpy as np


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


class BreastCancerRiskDataset(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        """
        Args:
            csv_file (str): Path to the CSV file containing patient data.
            image_dir (str): Directory containing mammogram images.
            transform (callable, optional): Optional transform to be applied on an image.
            n_years (int): Number of years for risk prediction (default: 5).
        """
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode

        # Ensure date columns in the CSV are in datetime format for efficient matching
        self.data["study_date_anon"] = pd.to_datetime(self.data["study_date_anon"], errors='coerce')

        self.image_data = self._load_image_data()
        self.longitudinal_samples = self._create_longitudinal_samples()

    def map_density(self, value):
        """Map density value to integer tensor suitable for GPU operations."""
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}  # Map 1-4
        index = mapping.get(value, -1)  # Use -1 for 'NA' if
        return torch.tensor(index, dtype=torch.long)

    def map_cancer_type(self, value):
        """Map cancer type (0,1,2) to integer tensor suitable for GPU ops.
           Return -1 if value is missing or outside {0,1,2}.
        """
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6}
        index = mapping.get(value, -1)
        return torch.tensor(index, dtype=torch.long)

    def _load_image_data(self):
        """
        REFACTORED: Load image data into a nested dictionary:
        { patient_id -> { date -> { view -> filename } } }
        This structure makes it easy to find all views for a given patient on a specific date.
        """
        image_data = defaultdict(lambda: defaultdict(dict))

        # Updated regex to be more robust
        pattern = r"(\d+)_([RL])_([A-Z]+)_(\d{4}-\d{2}-\d{2})_.*\.png"

        for filename in os.listdir(os.path.join(self.data_dir, self.mode).replace("\\", "/")):
            match = re.match(pattern, filename)
            if match:
                patient_id = match.group(1)
                laterality = match.group(2)
                view_type = match.group(3)  # CC or MLO
                date_str = match.group(4)

                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                full_view = f"{laterality}_{view_type}"  # e.g., "L_CC", "R_MLO"

                image_data[patient_id][date_obj][full_view] = filename
        return image_data

    def _create_longitudinal_samples(self):
        """
        Pre-compute longitudinal samples per patient.

        All pairs (training or testing) are shuffled so the model
        does not see them in consecutive order.
        """
        samples = []

        for patient_id, date_dict in self.image_data.items():
            if len(date_dict) < 2:
                continue

            sorted_dates = sorted(date_dict.keys())
            patient_pairs = []

            # Generate all consecutive pairs
            for idx in range(1, len(sorted_dates)):
                prior_date = sorted_dates[idx - 1]
                current_date = sorted_dates[idx]
                for laterality in ['L', 'R']:
                    cc_view = f"{laterality}_CC"
                    mlo_view = f"{laterality}_MLO"
                    if (cc_view in date_dict[current_date] and mlo_view in date_dict[current_date] and
                            cc_view in date_dict[prior_date] and mlo_view in date_dict[prior_date]):
                        patient_pairs.append((patient_id, laterality, current_date, prior_date))

            # Shuffle pairs for all modes
            if patient_pairs:
                random.shuffle(patient_pairs)

            # Add all pairs
            samples.extend(patient_pairs)
        random.shuffle(samples)
        print(f"Found {len(samples)} valid longitudinal multi-view samples in '{self.mode}' dataset.")
        return samples

    def __len__(self):
        return len(self.longitudinal_samples)

    def _get_metadata_for_image(self, filename):
        """Helper function to find the corresponding CSV row for a given image."""
        parts = filename.split("_")
        patient_id = parts[0]
        laterality = parts[1]
        view = parts[2]
        study_date = extract_date_from_filename(filename)

        matching_row = self.data[
            (self.data["patient_id"].astype(str) == patient_id) &
            (self.data["ImageLateralityFinal"] == laterality) &
            (self.data["view"] == view) &
            (self.data["study_date_anon"] == study_date)
            ]

        if matching_row.empty:
            return None  # Return None if no match is found
        return matching_row.iloc[0]  # Return the first matching row as a Series

    def __getitem__(self, idx):
        """
        REFACTORED: Fetches a complete longitudinal multi-view sample.
        Returns a dictionary containing the current and prior CC and MLO images.
        """
        # 1. Get the pre-computed sample information
        patient_id, laterality, current_date, prior_date = self.longitudinal_samples[idx]

        # 2. Retrieve all four filenames
        cc_view = f"{laterality}_CC"
        mlo_view = f"{laterality}_MLO"

        current_cc_file = self.image_data[patient_id][current_date][cc_view]
        current_mlo_file = self.image_data[patient_id][current_date][mlo_view]
        prior_cc_file = self.image_data[patient_id][prior_date][cc_view]
        prior_mlo_file = self.image_data[patient_id][prior_date][mlo_view]

        def load_and_process(filename):
            """Helper to load, convert, normalize, and transform an image."""
            img_path = os.path.join(self.data_dir, self.mode, filename)

            img_pil = Image.open(img_path)  # Ensure grayscale
            img_np = np.array(img_pil)

            img_tensor = torch.from_numpy(img_np).to(torch.float32)
            img_tensor = img_tensor / 65535.0
            if self.transform:
                img_tensor = self.transform(img_tensor)
                img_tensor = img_tensor.squeeze(0)
            else:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        current_image_cc = load_and_process(current_cc_file)
        current_image_mlo = load_and_process(current_mlo_file)
        previous_image_cc = load_and_process(prior_cc_file)
        previous_image_mlo = load_and_process(prior_mlo_file)
        if self.mode == 'train':
            images = torch.stack([current_image_cc, current_image_mlo, previous_image_cc, previous_image_mlo])
            if torch.rand(1).item() < 0.5:  # 50% probability of flipping
                images = torch.flip(images, dims=[-1])  # Flip horizontally
            current_image_cc, current_image_mlo, previous_image_cc, previous_image_mlo = images

        # 4. Calculate time gap
        time_gap = abs(current_date.year - prior_date.year)
        if time_gap > 5:
            time_gap = 5

        # 5. Fetch metadata (primarily based on the current screening date)
        # We use the CC view's metadata as the representative for this screening
        current_metadata = self._get_metadata_for_image(current_cc_file)
        if current_metadata is None:
            # Handle cases where metadata might be missing, return a dummy sample
            print(f"Warning: Could not find metadata for {current_cc_file}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self))

        # Process metadata to get target, mask, etc.
        years_last_followup = current_metadata["years_last_followup"]
        time_to_cancer = current_metadata["Time_to_Cancer_Years"]
        density = self.map_density(current_metadata["density"])
        cancer_type =  self.map_cancer_type(current_metadata["path_severity"])

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
            'current_image_cc': current_image_cc,
            'current_image_mlo': current_image_mlo,
            'previous_image_cc': previous_image_cc,
            'previous_image_mlo': previous_image_mlo,
            'time_gap': torch.tensor(time_gap, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'y_mask': torch.tensor(y_mask, dtype=torch.float32),
            'event_observed': torch.tensor(event_observed, dtype=torch.float32),
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density,
            'patient_id': patient_id,
            'cancer_type': cancer_type,
        }

        return data


