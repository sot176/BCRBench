# datasets/utils.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

def pad_to_length(values, pad_token: int, max_length: int) -> np.ndarray:
    values = list(values)[-max_length:]
    return np.array([pad_token] * (max_length - len(values)) + values)

def normalize_uint16(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return img / 65535.0


def load_image_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = np.array(img).astype(np.float32)

    img = normalize_uint16(img)

    tensor = torch.from_numpy(img).float()
    return tensor.unsqueeze(0)


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


def map_density(x):
    return torch.tensor({"A": 1, "B": 2, "C": 3}.get(x, -1))



def map_cancer_type(x):
    return torch.tensor({1: 1, 2: 2, 3: 3}.get(x, -1))


def load_tabular_data(csv_file):
    return pd.read_csv(csv_file, low_memory=False)


def build_row_lookup(df):
    lookup = {}

    for _, row in df.iterrows():

        key = (
            str(row["patient_id"]),
            row.get("laterality"),
            row.get("viewposition"),
            row.get("exam_year"),
        )

        lookup[key] = row

    return lookup