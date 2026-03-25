
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kornia.augmentation as K_A
from kornia.constants import Resample


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


class BreastCancerRiskDatasetCSAWCC_ImgFeatAlign(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index('file_name').to_dict(orient='index')

        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode
        self.image_data = self._load_image_data()
        self.patient_view_pairs = []

        for patient_id, view_images in self.image_data.items():
            if len(view_images) > 1:
                for i in range(len(view_images) - 1):
                    self.patient_view_pairs.append((patient_id, view_images, i))

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
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        index = mapping.get(value, -1)
        return torch.tensor(index, dtype=torch.long)
    def _load_image_data(self):
        """
        Load image data into a dictionary grouped by Patient ID, Laterality, and View.
        This ensures that prior and current images are from the same patient, same laterality, and same view.
        """
        image_data = defaultdict(list)
        csv_data = self.csv_data

        image_dir_path = os.path.join(self.data_dir, self.mode).replace("\\", "/")
        for filename in os.listdir(image_dir_path):
            if filename in csv_data:
                file_info = csv_data[filename]
                patient_id = str(file_info['patient_id'])
                laterality = file_info['laterality']  # Include laterality
                view = file_info['viewposition']
                exam_year = file_info['exam_year']
                years_to_cancer = file_info['years_to_cancer']
                years_last_followup = file_info['years_to_last_followup']
                density = file_info['density_group']
                cancer_type = file_info['x_type']
                # Group by (patient_id, laterality, view)
                image_data[(patient_id, laterality, view)].append(
                    (filename, exam_year, years_to_cancer, years_last_followup, density, cancer_type)
                )

        # Sort images by exam_year in ascending order (oldest first)
        for key in image_data:
            image_data[key].sort(key=lambda x: x[1])

        return image_data

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        if not self.patient_view_pairs:
            raise ValueError("No valid patient-view pairs generated.")
        if idx < 0 or idx >= len(self.patient_view_pairs):
            raise ValueError(f"Index {idx} is out of range. Valid range: 0 to {len(self.patient_view_pairs) - 1}")

        # Access patient-view pair
        patient_id, view_images, image_idx = self.patient_view_pairs[idx]

        # Ensure images are sorted by year (ascending)
        view_images.sort(key=lambda x: x[1])

        # Ensure we have at least two images per patient-view
        if len(view_images) < 2:
            raise ValueError(f"Not enough images for patient {patient_id} and view {view_images[0][1]}.")

        # Randomly pick one image (excluding the first one, to ensure a prior exists)
        selected_idx = random.randint(1, len(view_images) - 1)
        selected_image = view_images[selected_idx]  # Current image
        previous_image = view_images[selected_idx - 1]  # Prior image

        # Extract filenames and years
        current_image_file, current_year, time_to_cancer, years_last_followup, density, cancer_type = selected_image
        previous_image_file, previous_year, time_to_cancer_prior_img, years_last_followup_prior, _ , _= previous_image
        # Compute time gap
        time_gap = abs(current_year - previous_year)
        if time_gap > 5:  # Limit the time gap to 5 years max
            time_gap = 5

        # Load images
        current_image_path = os.path.join(self.data_dir, self.mode, current_image_file)
        previous_image_path = os.path.join(self.data_dir, self.mode, previous_image_file)

        current_image_pil = Image.open(current_image_path)
        current_image_pil.load()
        previous_image_pil = Image.open(previous_image_path)
        previous_image_pil.load()

        current_image_np = np.array(current_image_pil)
        current_image = (torch.from_numpy(current_image_np).to(torch.float16))
        previous_image_np = np.array(previous_image_pil)
        previous_image = (torch.from_numpy(previous_image_np).to(torch.float16))
        previous_image = previous_image.float() / 256.0
        current_image = current_image.float() / 256.0

        if self.transform is not None:
            current_image = self.transform(current_image)
            previous_image = self.transform(previous_image)
            previous_image = previous_image.squeeze(0)
            current_image = current_image.squeeze(0)

        else:
            previous_image = previous_image.unsqueeze(0)
            current_image = current_image.unsqueeze(0)

        density = self.map_density(density)
        cancer_type =  self.map_cancer_type(cancer_type)

        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):  # Check if the value is NaN
            time_to_cancer = 6

        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)
        time_to_cancer_prior_img = time_to_cancer + time_gap  # Adjust for the time gap

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


        else:
            # If no cancer, use the last follow-up year
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, 5))
            time_at_event = time_at_event - 1
            event_observed = 0
            time_at_event_prior = int(min(years_last_followup_prior, 5))
            time_at_event_prior = time_at_event_prior - 1


        y_mask = np.array([1] * (time_at_event + 1) + [0] * (5 - (time_at_event + 1)), dtype=np.int8)
        y_mask_prior = np.array(
            [1] * (time_at_event_prior + 1) + [0] * (5 - (time_at_event_prior + 1)), dtype=np.int8
        )
        y_mask = y_mask[:5]
        y_mask_prior = y_mask_prior[:5]
        # Create a mask for valid time points (before the last event or follow-up)

        return {
            'current_image': current_image,
            'previous_image': previous_image,
            'current_image_id': current_image_file,
            'previous_image_id': previous_image_file,
            'event_observed': event_observed,
            'event_times': time_at_event,
            'time_gap': time_gap,
            'y_mask': y_mask,
            'y_mask_prior': y_mask_prior,
            'target': target,
            'target_prior': target_prior,
            'density': density,
            'cancer_type': cancer_type,

        }


def main():
    path_data_train_val = "D:/UiT_PhD_datasets/csaw_dataset/Risk_dataset_train_val_test_1664_2048_new"
    csv_file = "D:/UiT PhD/CSAW/metadata_csawcc_dcm_path_new.csv"
    train_transform =torch.nn.Sequential(
        K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.05), degrees=0, shear=0, p=0.5),
        K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
        K_A.RandomGamma(gamma=(0.8, 1.4), gain=(0.95, 1.05), p=0.5),
        K_A.RandomCrop(size=(1843, 1498), p=0.2),
        K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
    )

    train_dataset = BreastCancerRiskDatasetCSAWCC(csv_file, path_data_train_val, 'train',
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