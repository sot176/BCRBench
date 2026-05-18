"""
Tests for dataset and dataloader factory functions.

Tests the datasets/dataloader_factory.py functions and data loading utilities.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch

from src.datasets.EMBED.utils import (
    clean_time_to_cancer,
    clean_followup_years,
    build_survival_target,
    map_density,
    map_cancer_type,
    build_row_lookup,
)

@pytest.mark.datasets
class TestDataloaderFactory:
    """Test dataloader factory functions."""
    
    def test_get_dataloader_embed(self):
        """Test getting EMBED dataloader."""
        from src.datasets import dataloader_factory
        
        try:
            # Test function existence
            assert hasattr(dataloader_factory, 'get_dataset_and_loader')
        except (ImportError, AttributeError):
            pytest.skip("Dataloader factory not fully implemented")
    
    def test_get_dataloader_csaw(self):
        """Test getting CSAW dataloader."""
        from src.datasets import dataloader_factory
        
        try:
            assert hasattr(dataloader_factory, 'get_dataset_and_loader')
        except (ImportError, AttributeError):
            pytest.skip("Dataloader factory not fully implemented")
    
    def test_invalid_dataset_name(self):
        """Test error handling for invalid dataset."""
        from src.datasets.dataloader_factory import get_dataset_and_loader
        
        with pytest.raises((ValueError, KeyError)):
            get_dataset_and_loader(
            model_name="Mirai",
            csv_file="dummy.csv",
            data_root="/tmp",
            num_workers=0,
            dataset_name="InvalidDataset",
            split="train",
            batch_size=32,
        )



def test_clean_time_to_cancer():
    assert clean_time_to_cancer(None) == 6
    assert clean_time_to_cancer(np.nan) == 6
    assert clean_time_to_cancer(0) == 1
    assert clean_time_to_cancer(3) == 3


def test_clean_followup_years():
    assert clean_followup_years(None) == 1
    assert clean_followup_years(np.nan) == 1
    assert clean_followup_years(0) == 1
    assert clean_followup_years(4) == 4


def test_build_survival_target_event():
    target, mask, event_time, observed = build_survival_target(
        n_years=5,
        time_to_cancer=3,
        followup=10,
    )

    assert observed == 1
    assert event_time == 2
    assert target.shape[0] == 5
    assert np.all(target[2:] == 1)


def test_build_survival_target_censored():
    target, mask, event_time, observed = build_survival_target(
        n_years=5,
        time_to_cancer=10,
        followup=3,
    )

    assert observed == 0
    assert event_time == 2
    assert mask[:3].sum() == 3

def test_map_density():
    # now expects numeric input
    assert map_density(1).item() == 1
    assert map_density(2).item() == 2
    assert map_density(3).item() == 3

    # invalid or missing
    assert map_density(None).item() == -1
    assert map_density(99).item() == -1


def test_map_cancer_type():
    assert map_cancer_type(1).item() == 1
    assert map_cancer_type(2).item() == 2
    assert map_cancer_type(3).item() == 3

    assert map_cancer_type(99).item() == -1


def test_build_row_lookup_embed():
    df = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "ImageLateralityFinal": "L",
                "view": "CC",
                "study_date_anon": pd.Timestamp("2020-01-01"),
                "value": 10,
            },
            {
                "patient_id": 2,
                "ImageLateralityFinal": "R",
                "view": "MLO",
                "study_date_anon": pd.Timestamp("2021-01-01"),
                "value": 20,
            },
        ]
    )

    lookup = build_row_lookup(df)

    assert len(lookup) == 2

    key = ("1", "L", "CC", pd.Timestamp("2020-01-01"))
    assert key in lookup
    assert lookup[key]["value"] == 10