
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
from  utils import RACE_TO_ID

def imgunit16(img):
    mammogram_scaled = (
        (img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 65535
    )
    return mammogram_scaled


class BreastCancerRiskDatasetEMBED_ImgFeatAlign(Dataset):
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
        self.image_data = self._load_image_data()
        # Precompute the patient-view pairs once during initialization
        self.patient_view_pairs = []

        for patient_id, view_images in self.image_data.items():
            if (
                    len(view_images) > 1
            ):  # Only consider patient-view pairs with multiple images
                for i in range(len(view_images) - 1):
                    self.patient_view_pairs.append((patient_id, view_images, i))

        # Store the length of the dataset (number of valid patient-view pairs)
        self.num_elements = len(self.patient_view_pairs)

    def map_density(self, value):
        if value == 1:
            return "A"
        elif value == 2:
            return "B"
        elif value == 3:
            return "C"
        elif value == 4:
            return "D"
        else:
            return "NA"
    def map_cancer_type(self, value):
        """Map cancer type (0,1,2) to integer tensor suitable for GPU ops.
           Return -1 if value is missing or outside {0,1,2}.
        """
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6}
        index = mapping.get(value, -1)
        return torch.tensor(index, dtype=torch.long)

    def _load_image_data(self):
        """
        Load image data into a dictionary grouped by patient ID and view type.
        """
        image_data = defaultdict(list)

        # Regex pattern to extract the necessary information from the filename
        pattern = r"(\d+)_([A-Z]+_[A-Z]+)_(\d{4})-\d{2}-\d{2}_.*\.png"

        # Iterate over the files in the image directory
        for filename in os.listdir(
                os.path.join(self.data_dir, self.mode).replace("\\", "/")
        ):
            # Match the filename to extract the relevant info
            match = re.match(pattern, filename)
            if match:
                patient_id = match.group(1)
                view = match.group(2)
                year = int(match.group(3))

                # Store the filename with patient_id and view as the key
                image_data[(patient_id, view)].append((filename, year))

        return image_data

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        if not self.patient_view_pairs:
            raise ValueError("No valid patient-view pairs generated.")
        if idx < 0 or idx >= len(self.patient_view_pairs):
            raise ValueError(
                f"Index {idx} is out of range. Valid range: 0 to {len(self.patient_view_pairs) - 1}"
            )

            # Access the patient-view pair
        patient_id, view_images, image_idx = self.patient_view_pairs[idx]

        # Extract the date from the filename
        def extract_date_from_filename(filename):
            date_str = filename.split("_")[3]  # e.g., "2019-07-13"
            return datetime.strptime(date_str, "%Y-%m-%d")  # Convert to datetime object

        # Sort the images by year (ascending order)

        view_images.sort(key=lambda x: x[1])

        # Ensure there are at least two images to form a valid current-previous pair
        if len(view_images) < 2:
            raise ValueError(
                f"Not enough images for patient {patient_id} and view {view_images[0][1]}."
            )

        # Randomly pick one image, and then get its previous image
        idx = random.randint(
            1, len(view_images) - 1
        )  # Ensure at least one previous image exists
        selected_image = view_images[
            idx
        ]  # Randomly chosen image (e.g., the current image)
        previous_image = view_images[idx - 1]  # Previous image

        # Extract the filenames
        current_image_file = selected_image[0]
        previous_image_file = previous_image[0]
        current_year = extract_date_from_filename(current_image_file)
        previous_year = extract_date_from_filename(previous_image_file)

        time_gap = abs(current_year.year - previous_year.year)
        if time_gap > 5:
            time_gap = 5
        current_image_path = os.path.join(self.data_dir, self.mode, current_image_file)
        previous_image_path = os.path.join(
            self.data_dir, self.mode, previous_image_file
        )
        try:
            current_image_pil = Image.open(current_image_path)
            current_image_pil.load()
            previous_image_pil = Image.open(previous_image_path)
            previous_image_pil.load()
        except OSError:
            print(f"Problematic current image: {current_image_path}")
            print(f"Problematic previous image: {previous_image_path}")


        current_image_np = np.array(current_image_pil)
        current_image = (
            torch.from_numpy(current_image_np).to(torch.float16)
        )
        previous_image_np = np.array(previous_image_pil)

        previous_image = (
            torch.from_numpy(previous_image_np).to(torch.float16)
        )
        previous_image = previous_image.float() / 65535.0
        current_image = current_image.float() / 65535.0
        if self.transform is not None:
            current_image = self.transform(current_image)
            previous_image = self.transform(previous_image)
            previous_image = previous_image.squeeze(0)
            current_image = current_image.squeeze(0)

        else:
            previous_image = previous_image.unsqueeze(0)
            current_image = current_image.unsqueeze(0)

        # Extract patient ID, laterality, and view from the filename
        parts = current_image_file.split("_")
        patient_id = parts[0]
        image_laterality = parts[1]
        view = parts[2]
        parts_prior = previous_image_file.split("_")
        patient_id_prior = parts_prior[0]
        image_laterality_prior = parts_prior[1]
        view_prior = parts_prior[2]

        # Ensure study_date and data['study_date_anon'] are datetime objects
        self.data["study_date_anon"] = pd.to_datetime(
            self.data["study_date_anon"], format="%Y-%m-%d"
        )
        study_date = pd.to_datetime(current_year, format="%Y-%m-%d")
        study_date_prior = pd.to_datetime(previous_year, format="%Y-%m-%d")
        # Find the corresponding row in the CSV
        matching_row = self.data[
            (self.data["patient_id"].astype(str) == str(patient_id))
            & (self.data["ImageLateralityFinal"].astype(str) == str(image_laterality))
            & (self.data["view"].astype(str) == str(view))
            & (self.data["study_date_anon"] == study_date)
            ]

        matching_row_prior = self.data[
            (self.data["patient_id"].astype(str) == str(patient_id_prior))
            & (
                    self.data["ImageLateralityFinal"].astype(str)
                    == str(image_laterality_prior)
            )
            & (self.data["view"].astype(str) == str(view_prior))
            & (self.data["study_date_anon"] == study_date_prior)
            ]
        # Initialize variables
        years_last_followup_prior = matching_row_prior["years_last_followup"].values[0]
        years_last_followup = matching_row["years_last_followup"].values[0]

        time_to_cancer = matching_row["Time_to_Cancer_Years"].values[0]
        time_to_cancer_prior_img = matching_row_prior["Time_to_Cancer_Years"].values[0]
        density = matching_row["density"].values[0]
        density = self.map_density(density)
        cancer_type =  self.map_cancer_type(matching_row["path_severity"].values[0])
        race_col = matching_row.get("RACE_DESC")

        if race_col is None or len(matching_row) == 0:
            race_str = "Unknown"
        else:
            race = race_col.values[0]
            if not isinstance(race, str) or race.strip() == "":
                race_str = "Unknown"
            else:
                race_str = race.strip()

        race_id = RACE_TO_ID.get(race_str, RACE_TO_ID["Unknown"])

        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):  # Check if the value is NaN
            time_to_cancer = 6
        if time_to_cancer_prior_img == 0:
            time_to_cancer_prior_img = 1
        if pd.isna(time_to_cancer_prior_img):  # Check if the value is NaN
            time_to_cancer_prior_img = 6

        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)
        time_to_cancer_prior_img = int(time_to_cancer_prior_img)

        time_to_cancer = time_to_cancer - 1

        time_to_cancer_prior_img = time_to_cancer_prior_img - 1
        any_cancer = time_to_cancer < 5

        # Initialize the sequence with zeros
        target = np.zeros(5, dtype=np.int8)
        target_prior = np.zeros(5, dtype=np.int8)

        # If the patient has cancer, mark the event year and later with 1
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            time_at_event_prior = int(time_to_cancer_prior_img)
            target[time_at_event:] = 1
            target_prior[time_at_event_prior:] = 1
            # target[-1] = 0
            # target_prior[-1] = 0

        else:
            # If no cancer, use the last follow-up year
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, 5))
            time_at_event = time_at_event - 1
            event_observed = 0
            time_at_event_prior = int(min(years_last_followup_prior, 5))
            time_at_event_prior = time_at_event_prior - 1
            # target[-1] = 1
            # target_prior[-1] = 1

        y_mask = np.array([1] * (time_at_event + 1) + [0] * (5 - (time_at_event + 1)), dtype=np.int8)
        y_mask_prior = np.array(
            [1] * (time_at_event_prior + 1) + [0] * (5 - (time_at_event_prior + 1)), dtype=np.int8
        )
        y_mask = y_mask[:5]
        y_mask_prior = y_mask_prior[:5]
        # Create a mask for valid time points (before the last event or follow-up)
        event_times = time_at_event
        data = {
            'current_image': current_image,
            'previous_image': previous_image,
            'current_image_id': current_image_file,
            'previous_image_id': previous_image_file,
            'event_observed': event_observed,
            'event_times': event_times,
            'time_gap': time_gap,
            'y_mask': y_mask,
            'y_mask_prior': y_mask_prior,
            'years_to_cancer': time_to_cancer,
            'years_to_cancer_prior': time_to_cancer_prior_img,
            'years_to_last_followup': years_last_followup,
            'years_to_last_followup_prior': years_last_followup_prior,
            'target': target,
            'target_prior': target_prior,
            'density': density,
            'cancer_type': cancer_type,
            'race':  torch.tensor(race_id, dtype=torch.long),

        }
        return data


def main():
    path_data_train_val = "C:/UiT_PhD_datasets/embed_dataset/risk_dataset_1664_2048/"
    #csv_file = "D:/UiT PhD/EMBED/combined_cases_with_followup_new.csv"
    csv_file = "C:/Users/sothr1456/OneDrive - UiT Office 365/Documents/UiT PhD/EMBED/combined_cases_with_followup_race.csv"
    train_transform  =torch.nn.Sequential(
        K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.05), degrees=0, shear=0, p=0.5),
        K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
        K_A.RandomGamma(gamma=(0.8, 1.4), gain=(0.95, 1.05), p=0.5),
        K_A.RandomCrop(size=(1843, 1498), p=0.2),
        K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
    )

    train_dataset = BreastCancerRiskDataset(csv_file, path_data_train_val, 'train',
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
    print("cancer type ", cancer_type)
    # Compute class weights
    classcount = torch.tensor([539, 190, 163, 90, 67], dtype=torch.float)

    n_samples = classcount.sum()
    n_classes = len(classcount)

    class_weights = n_samples / (n_classes * classcount)
    print(class_weights)
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
        print(img_current.shape)
        print(img_current.dtype)
        current_image_float32 = img_current[i].squeeze()#.cpu().numpy()  # Convert to float32 and move to CPU
        previous_image_float32 = img_previous[i].squeeze()#.cpu().numpy()  # Same for previous image

        plt.subplot(1, 2, 1)
        plt.imshow(current_image_float32, cmap='gray')
        plt.title('Current Image')
        # Plot the previous image in the second subplot (1 row, 2 columns, 2nd position)
        plt.subplot(1, 2, 2)
        plt.imshow(previous_image_float32, cmap='gray')
        plt.title('Previous Image')
        plt.show()
    label = batch['label']
    print(label)

if __name__ == '__main__':
    main()