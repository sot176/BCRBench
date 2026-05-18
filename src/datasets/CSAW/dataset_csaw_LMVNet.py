from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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

@dataclass(frozen=True)
class CSAWLongitudinalSample:
    patient_id: str
    laterality: str
    current_year: int
    prior_year: int


class BreastCancerRiskDatasetCSAWCCLMVNet(Dataset):
    """
    Longitudinal CSAW dataset for paired current/prior CC+MLO exams.

    Each sample contains:
    - current CC/MLO for one laterality
    - previous CC/MLO for the same laterality
    - survival target built from the current exam
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
        self.samples = self._build_longitudinal_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range.")

        sample = self.samples[index]

        current_exam = self.image_data[sample.patient_id][sample.current_year]
        prior_exam = self.image_data[sample.patient_id][sample.prior_year]

        cc_key = f"{sample.laterality}_CC"
        mlo_key = f"{sample.laterality}_MLO"

        cur_cc_record = current_exam[cc_key]
        cur_mlo_record = current_exam[mlo_key]
        pri_cc_record = prior_exam[cc_key]
        pri_mlo_record = prior_exam[mlo_key]

        current_image_cc = load_image_tensor(
            self.data_dir / self.mode / cur_cc_record.filename
        )
        current_image_mlo = load_image_tensor(
            self.data_dir / self.mode / cur_mlo_record.filename
        )
        previous_image_cc = load_image_tensor(
            self.data_dir / self.mode / pri_cc_record.filename
        )
        previous_image_mlo = load_image_tensor(
            self.data_dir / self.mode / pri_mlo_record.filename
        )

        if self.transform:
            current_image_cc = self.transform(current_image_cc).squeeze(0)
            current_image_mlo = self.transform(current_image_mlo).squeeze(0)
            previous_image_cc = self.transform(previous_image_cc).squeeze(0)
            previous_image_mlo = self.transform(previous_image_mlo).squeeze(0)

        if self.mode == "train" and torch.rand(1).item() < 0.5:
            stacked = torch.stack([
                current_image_cc,
                current_image_mlo,
                previous_image_cc,
                previous_image_mlo,
            ])
            stacked = torch.flip(stacked, dims=[-1])
            (
                current_image_cc,
                current_image_mlo,
                previous_image_cc,
                previous_image_mlo,
            ) = stacked

        curr_row = self._lookup_row(cur_cc_record)

        curr_ttc = clean_time_to_cancer(curr_row["years_to_cancer"])
        curr_fu = clean_followup_years(curr_row["years_to_last_followup"])

        target, y_mask, event_time, event_observed = build_survival_target(
            self.n_years,
            curr_ttc,
            curr_fu,
        )

        time_gap = min(
            abs(sample.current_year - sample.prior_year),
            self.n_years,
        )

        return {
            "current_image_cc": current_image_cc,
            "current_image_mlo": current_image_mlo,
            "previous_image_cc": previous_image_cc,
            "previous_image_mlo": previous_image_mlo,
            "patient_id": sample.patient_id,
            "current_image_id_cc": cur_cc_record.filename,
            "current_image_id_mlo": cur_mlo_record.filename,
            "previous_image_id_cc": pri_cc_record.filename,
            "previous_image_id_mlo": pri_mlo_record.filename,
            "time_gap": time_gap,
            "target": target,
            "y_mask": y_mask,
            "event_observed": event_observed,
            "event_times": event_time,
            "years_to_cancer": curr_ttc - 1,
            "years_to_last_followup": curr_fu,
            "density": map_density(curr_row["density_group"]),
            "cancer_type": map_cancer_type(curr_row["x_type"]),
        }

    def _build_row_lookup(self) -> dict[str, dict]:
        return self.data.set_index("file_name").to_dict(orient="index")

    def _load_image_data(
        self,
    ) -> dict[str, dict[int, dict[str, CSAWImageRecord]]]:
        image_data = defaultdict(lambda: defaultdict(dict))
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
            exam_key = f"{laterality}_{view}"

            image_data[patient_id][exam_year][exam_key] = CSAWImageRecord(
                filename=image_path.name,
                patient_id=patient_id,
                exam_year=exam_year,
                laterality=laterality,
                view=view,
            )

        return image_data

    def _build_longitudinal_samples(self) -> list[CSAWLongitudinalSample]:
        samples: list[CSAWLongitudinalSample] = []

        for patient_id, exams_by_year in self.image_data.items():
            sorted_years = sorted(exams_by_year.keys())
            if len(sorted_years) < 2:
                continue

            for i in range(1, len(sorted_years)):
                prior_year = sorted_years[i - 1]
                current_year = sorted_years[i]

                for laterality in ["L", "R"]:
                    cc_key = f"{laterality}_CC"
                    mlo_key = f"{laterality}_MLO"

                    if not (
                        cc_key in exams_by_year[current_year]
                        and mlo_key in exams_by_year[current_year]
                        and cc_key in exams_by_year[prior_year]
                        and mlo_key in exams_by_year[prior_year]
                    ):
                        continue

                    samples.append(
                        CSAWLongitudinalSample(
                            patient_id=patient_id,
                            laterality=laterality,
                            current_year=current_year,
                            prior_year=prior_year,
                        )
                    )

        random.shuffle(samples)
        print(f"Found {len(samples)} CSAW longitudinal samples in '{self.mode}'")
        return samples

    def _lookup_row(self, record: CSAWImageRecord) -> dict:
        try:
            return self.row_lookup[record.filename]
        except KeyError as exc:
            raise KeyError(
                f"No CSV row matched image {record.filename}."
            ) from exc
 