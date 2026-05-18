from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
 
from utils import (
    load_tabular_data,
    load_image_tensor,
    build_survival_target,
    clean_time_to_cancer,
    clean_followup_years,
    map_density,
    map_cancer_type,
)

 
@dataclass(frozen=True)
class CSAWImageRecord:
    filename: str
    patient_id: str
    exam_year: int
    laterality: str
    view: str


class BreastCancerRiskDatasetCSAWCCImgFeatAlign(Dataset):
    """
    Longitudinal CSAW mammography dataset.

    Each sample contains a current mammogram and the previous mammogram
    for the same patient/laterality/view pair, along with survival targets.
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
        self.image_pairs = self._build_image_pairs()

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(f"Index {index} out of range.")

        prev_record, curr_record = self.image_pairs[index]

        current_image = load_image_tensor(
            self.data_dir / self.mode / curr_record.filename
        )
        previous_image = load_image_tensor(
            self.data_dir / self.mode / prev_record.filename
        )

        if self.transform:
            current_image = self.transform(current_image).squeeze(0)
            previous_image = self.transform(previous_image).squeeze(0)

        curr_row = self._lookup_row(curr_record)
        prev_row = self._lookup_row(prev_record)

        curr_ttc = clean_time_to_cancer(curr_row["years_to_cancer"])
        prev_ttc = clean_time_to_cancer(prev_row["years_to_cancer"])

        curr_fu = clean_followup_years(curr_row["years_to_last_followup"])
        prev_fu = clean_followup_years(prev_row["years_to_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            curr_ttc,
            curr_fu,
        )

        prev_target, prev_y_mask, prev_event_time, _ = build_survival_target(
            self.n_years,
            prev_ttc,
            prev_fu,
        )

        time_gap = min(abs(curr_record.exam_year - prev_record.exam_year), self.n_years)

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": curr_record.filename,
            "previous_image_id": prev_record.filename,
            "event_observed": event_observed,
            "event_times": event_time,
            "previous_event_times": prev_event_time,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": prev_y_mask,
            "target": target,
            "target_prior": prev_target,
            "years_to_cancer": curr_ttc - 1,
            "years_to_cancer_prior": prev_ttc - 1,
            "years_to_last_followup": curr_fu,
            "years_to_last_followup_prior": prev_fu,
            "density": map_density(curr_row["density_group"]),
            "cancer_type": map_cancer_type(curr_row["x_type"]),
        }

    def _build_row_lookup(self) -> dict[str, dict]:
        return self.data.set_index("file_name").to_dict(orient="index")

    def _load_image_data(self) -> dict[tuple[str, str, str], list[CSAWImageRecord]]:
        image_data: dict[tuple[str, str, str], list[CSAWImageRecord]] = defaultdict(list)
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

            image_data[(record.patient_id, record.laterality, record.view)].append(record)

        return image_data

    def _build_image_pairs(self) -> list[tuple[CSAWImageRecord, CSAWImageRecord]]:
        pairs: list[tuple[CSAWImageRecord, CSAWImageRecord]] = []

        for records in self.image_data.values():
            records = sorted(records, key=lambda record: record.exam_year)
            pairs.extend(zip(records[:-1], records[1:]))

        print(f"Found {len(pairs)} image pairs in {self.mode}")
        return pairs

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        try:
            return self.row_lookup[record.filename]
        except KeyError as exc:
            raise KeyError(
                f"No CSV row matched image {record.filename}."
            ) from exc
 