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


@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    laterality: str
    view: str
    study_date: pd.Timestamp


@dataclass(frozen=True)
class CSAWPairSample:
    patient_id: str
    current_date: pd.Timestamp
    prior_date: pd.Timestamp
    current_views: dict[str, CSAWImageRecord]
    prior_views: dict[str, CSAWImageRecord]


def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)

class BreastCancerRiskDatasetCSAWVMRA(Dataset):

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

        # CURRENT EXAM
        current_left_cc = sample.current_views["L_CC"]
        current_left_mlo = sample.current_views["L_MLO"]

        current_right_cc = sample.current_views["R_CC"]
        current_right_mlo = sample.current_views["R_MLO"]

        # PRIOR EXAM
        prior_left_cc = sample.prior_views["L_CC"]
        prior_left_mlo = sample.prior_views["L_MLO"]

        prior_right_cc = sample.prior_views["R_CC"]
        prior_right_mlo = sample.prior_views["R_MLO"]

        # LOAD CURRENT
        current_images = torch.stack([
            self._load_image(current_left_cc.filename),
            self._load_image(current_left_mlo.filename),
            self._load_image(current_right_cc.filename),
            self._load_image(current_right_mlo.filename),
        ])

        # LOAD PRIOR
        prior_images = torch.stack([
            self._load_image(prior_left_cc.filename),
            self._load_image(prior_left_mlo.filename),
            self._load_image(prior_right_cc.filename),
            self._load_image(prior_right_mlo.filename),
        ])

        if self.transform is not None:

            current_images = torch.stack([
                self.transform(img).squeeze(0)
                for img in current_images
            ])

            prior_images = torch.stack([
                self.transform(img).squeeze(0)
                for img in prior_images
            ])

        images = torch.stack([prior_images, current_images])

        current_row = self.csv_data[
            current_right_cc.filename
        ]

        prior_row = self.csv_data[
            prior_right_cc.filename
        ]

        current_ttc = self._clean_time(
            current_row["years_to_cancer"]
        )

        prior_ttc = self._clean_time(
            prior_row["years_to_cancer"]
        )

        current_followup = self._clean_followup(
            current_row["years_to_last_followup"]
        )

        prior_followup = self._clean_followup(
            prior_row["years_to_last_followup"]
        )

        target, y_mask, event_time, event_observed = (
            self._build_survival_target(
                current_ttc,
                current_followup,
            )
        )

        target_prior, y_mask_prior, event_time_prior, _ = (
            self._build_survival_target(
                prior_ttc,
                prior_followup,
            )
        )

        time_gap = min(
            abs(
                sample.current_date.year
                - sample.prior_date.year
            ),
            self.n_years,
        )
        all_views = [0, 1, 0, 1]
        all_sides = [1, 1, 0, 0]
        all_times = [0, 0, 0, 0]

        time_seq = pad_to_length(all_times, MAX_TIME, len(all_times))
        view_seq = pad_to_length(all_views, MAX_VIEWS, len(all_views))
        side_seq = pad_to_length(all_sides, MAX_SIDES, len(all_sides))
        
        exam_mask = torch.ones(2, dtype=torch.bool) 
        view_mask = torch.ones(2, 4, dtype=torch.bool)

        return {
            'images': images,
            'exam_mask': exam_mask,
            'view_mask': view_mask,
            "target": torch.tensor( target, dtype=torch.float32,),
            "y_mask": torch.tensor( y_mask,dtype=torch.float32,),
            "event_observed": torch.tensor( event_observed,dtype=torch.float32,),
            "event_times": torch.tensor(event_time,dtype=torch.float32,),
            "time_gap": torch.tensor(time_gap,dtype=torch.float32,),
            "density": self._map_density(current_row["density_group"]),
            "cancer_type": self._map_cancer_type(current_row["x_type"]),
            "patient_id": sample.patient_id,
            "time_seq": torch.tensor(time_seq,dtype=torch.long, ),
            "view_seq": torch.tensor(view_seq,dtype=torch.long,),
            "side_seq": torch.tensor(side_seq,dtype=torch.long,),
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

        # build complete exams first
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

        patient_exams = defaultdict(list)

        for (patient_id, study_date), views in exams.items():

            if required_views.issubset(views.keys()):

                patient_exams[patient_id].append(
                    (study_date, views)
                )

        samples = []

        for patient_id, exam_list in patient_exams.items():

            exam_list = sorted(
                exam_list,
                key=lambda x: x[0]
            )

            if len(exam_list) < 2:
                continue

            for i in range(1, len(exam_list)):

                prior_date, prior_views = exam_list[i - 1]

                current_date, current_views = exam_list[i]

                samples.append(
                    CSAWPairSample(
                        patient_id=patient_id,
                        current_date=current_date,
                        prior_date=prior_date,
                        current_views=current_views,
                        prior_views=prior_views,
                    )
                )

        print(f"[INFO] Found {len(samples)} longitudinal pairs")

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

