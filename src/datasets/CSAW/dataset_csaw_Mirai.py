import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset 
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset 
 

MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=20

"""
Images are stored as PNG but keep the same base name from the original DICOM file
00004_20990909_R_MLO_2.dcm  →  00004_20990909_R_MLO_2.png
The CSV must contain the file_name e.g. 00004_20990909_R_MLO_2.png
Each row corresponds to exactly ONE IMAGE.
Each exam has 4 images: L_CC, L_MLO, R_CC, R_MLO
"""


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

class BreastCancerRiskDatasetCSAWMirai(Dataset):

    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index("file_name").to_dict(orient="index")

        self.data_dir = Path(image_dir)
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
 
        images = torch.stack(
            [
                left_cc_image,
                left_mlo_image,
                right_cc_image,
                right_mlo_image,
            ]
        )
 
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
                laterality=("L"if meta["laterality"] == "Left" else "R" ),
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


 