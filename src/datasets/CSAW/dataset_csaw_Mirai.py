from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import (
    load_tabular_data,
    build_survival_target,
    clean_time_to_cancer,
    clean_followup_years,
    map_density,
    map_cancer_type,
    pad_to_length
)


MAX_TIME = 20
MAX_VIEWS = 2
MAX_SIDES = 2


@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    exam_year: int
    laterality: str
    view: str


@dataclass(frozen=True)
class CSAWExamSample:
    patient_id: str
    exam_year: int
    views: dict[str, CSAWImageRecord]
 
class BreastCancerRiskDatasetCSAWMirai(Dataset):
    """
    Exam-level CSAW dataset for MIRAI-style four-view input.

    Each sample contains one complete exam:
    - L_CC
    - L_MLO
    - R_CC
    - R_MLO
    """

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        mode: str,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_years: int = 5,
    ) -> None:
        self.data_dir = Path(image_dir)
        self.mode = mode
        self.transform = transforms
        self.n_years = n_years

        self.data = load_tabular_data(csv_file)
        self.row_lookup = self._build_row_lookup()

        self.image_data = self._load_image_data()
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range.")

        sample = self.samples[index]

        left_cc = sample.views["L_CC"]
        left_mlo = sample.views["L_MLO"]
        right_cc = sample.views["R_CC"]
        right_mlo = sample.views["R_MLO"]

        left_cc_image = self._load_image(left_cc.filename)
        left_mlo_image = self._load_image(left_mlo.filename)
        right_cc_image = self._load_image(right_cc.filename)
        right_mlo_image = self._load_image(right_mlo.filename)

        if self.transform is not None:
            left_cc_image = self.transform(left_cc_image).squeeze(0)
            left_mlo_image = self.transform(left_mlo_image).squeeze(0)
            right_cc_image = self.transform(right_cc_image).squeeze(0)
            right_mlo_image = self.transform(right_mlo_image).squeeze(0)

        images = torch.stack(
            [left_cc_image, left_mlo_image, right_cc_image, right_mlo_image]
        )

        curr_row = self._lookup_row(right_cc)

        current_time_to_cancer = clean_time_to_cancer(curr_row["years_to_cancer"])
        years_last_followup = clean_followup_years(curr_row["years_to_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            current_time_to_cancer,
            years_last_followup,
        )

        all_views = [0, 1, 0, 1]
        all_sides = [0, 0, 1, 1]
        all_times = [0, 0, 0, 0]

        time_seq = pad_to_length(all_times, pad_token=MAX_TIME, max_length=len(all_times))
        view_seq = pad_to_length(all_views, pad_token=MAX_VIEWS, max_length=len(all_views))
        side_seq = pad_to_length(all_sides, pad_token=MAX_SIDES, max_length=len(all_sides))

        return {
            "images": images,
            "target": target,
            "y_mask": y_mask,
            "event_observed": event_observed,
            "event_times": event_time,
            "density": map_density(curr_row["density_group"]),
            "cancer_type": map_cancer_type(curr_row["x_type"]),
            "patient_id": sample.patient_id,
            "exam_year": sample.exam_year,
            "image_ids": [
                left_cc.filename,
                left_mlo.filename,
                right_cc.filename,
                right_mlo.filename,
            ],
            "time_seq": torch.tensor(time_seq, dtype=torch.long),
            "view_seq": torch.tensor(view_seq, dtype=torch.long),
            "side_seq": torch.tensor(side_seq, dtype=torch.long),
            "years_to_cancer": current_time_to_cancer - 1,
            "years_to_last_followup": years_last_followup,
        }

    def _build_row_lookup(self) -> dict[str, dict]:
        return self.data.set_index("file_name").to_dict(orient="index")

    def _load_image_data(
        self,
    ) -> dict[tuple[str, int], dict[str, CSAWImageRecord]]:
        image_data: dict[tuple[str, int], dict[str, CSAWImageRecord]] = defaultdict(dict)
        image_folder = self.data_dir / self.mode

        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        for image_path in image_folder.iterdir():
            row = self.row_lookup.get(image_path.name)
            if row is None:
                continue

            patient_id = str(row["patient_id"])
            exam_year = int(row["exam_year"])
            laterality = "L" if row["laterality"] == "Left" else "R"
            view = row["viewposition"]

            record = CSAWImageRecord(
                filename=image_path.name,
                patient_id=patient_id,
                exam_year=exam_year,
                laterality=laterality,
                view=view,
            )

            image_data[(patient_id, exam_year)][f"{laterality}_{view}"] = record

        return image_data

    def _build_samples(self) -> list[CSAWExamSample]:
        required_views = {"L_CC", "L_MLO", "R_CC", "R_MLO"}
        samples: list[CSAWExamSample] = []

        for (patient_id, exam_year), views in self.image_data.items():
            if not required_views.issubset(views.keys()):
                continue

            samples.append(
                CSAWExamSample(
                    patient_id=patient_id,
                    exam_year=exam_year,
                    views=views,
                )
            )

        print(f"[INFO] Found {len(samples)} complete exams")
        return samples

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        try:
            return self.row_lookup[record.filename]
        except KeyError as exc:
            raise KeyError(f"No CSV row matched image {record.filename}.") from exc

     
 