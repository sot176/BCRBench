import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime


def load_tabular_data(csv_file: str | Path) -> pd.DataFrame:
    data = pd.read_csv(csv_file, low_memory=False)
    data["study_date_anon"] = pd.to_datetime(data["study_date_anon"], format="%Y-%m-%d")
    return data


def build_row_lookup(data: pd.DataFrame):
    lookup = {}
    for _, row in data.iterrows():
        key = (
            str(row["patient_id"]),
            str(row["ImageLateralityFinal"]),
            str(row["view"]),
            row["study_date_anon"],
        )
        lookup[key] = row
    return lookup


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.array(img)

    tensor = torch.from_numpy(arr).float() / 65535.0
    return tensor.unsqueeze(0)


def map_density(value) -> torch.Tensor:
    value = -1 if pd.isna(value) else int(value)
    return torch.tensor(value if value in {1, 2, 3, 4} else -1, dtype=torch.long)


def map_cancer_type(value) -> torch.Tensor:
    value = -1 if pd.isna(value) else int(value)
    return torch.tensor(value if value in set(range(7)) else -1, dtype=torch.long)


def map_race(value, race_to_id) -> torch.Tensor:
    race = value.strip() if isinstance(value, str) and value.strip() else "Unknown"
    return torch.tensor(race_to_id.get(race, race_to_id["Unknown"]), dtype=torch.long)


def clean_time_to_cancer(value) -> int:
    if pd.isna(value):
        return 6
    value = int(value)
    return 1 if value == 0 else value


def clean_followup_years(value) -> int:
    if pd.isna(value):
        return 1
    value = int(value)
    return 1 if value == 0 else value


def build_survival_target(n_years: int, time_to_cancer: int, followup: int):
    target = np.zeros(n_years, dtype=np.int8)

    event_time = time_to_cancer - 1
    event_observed = int(event_time < n_years)

    if event_observed:
        target[event_time:] = 1
    else:
        event_time = min(followup, n_years) - 1

    y_mask = np.zeros(n_years, dtype=np.int8)
    y_mask[: event_time + 1] = 1

    return target, y_mask, event_time, event_observed


def scale_to_uint16_range(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 65535.0