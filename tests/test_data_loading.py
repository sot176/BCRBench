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


@pytest.mark.datasets
class TestDataBatch:
    """Test data batch structure and properties."""
    
    def test_batch_has_required_fields(self, sample_batch):
        """Test that batch has required fields."""
        required_fields = ["images", "labels"]
        for field in required_fields:
            assert field in sample_batch
    
    def test_batch_labels_shape(self, sample_batch):
        """Test batch labels shape."""
        labels = sample_batch["labels"]
        assert len(labels.shape) >= 1
        assert labels.shape[0] == sample_batch["images"].shape[0]
    
    def test_batch_tensor_types(self, sample_batch):
        """Test that batch contains tensors."""
        assert isinstance(sample_batch["images"], torch.Tensor)
        assert isinstance(sample_batch["labels"], torch.Tensor)

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
    assert event_time == 2  # 3 - 1
    assert target.shape[0] == 5
    assert np.all(target[2:] == 1)


def test_build_survival_target_censored():
    target, mask, event_time, observed = build_survival_target(
        n_years=5,
        time_to_cancer=10,
        followup=3,
    )

    assert observed == 0
    assert event_time == 2  # followup - 1
    assert mask[:3].sum() == 3

def test_map_density():
    assert map_density("A").item() == 1
    assert map_density("B").item() == 2
    assert map_density("C").item() == 3
    assert map_density("X").item() == -1


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
                "laterality": "L",
                "viewposition": "CC",
                "exam_year": 2020,
                "value": 10,
            },
            {
                "patient_id": 2,
                "laterality": "R",
                "viewposition": "MLO",
                "exam_year": 2021,
                "value": 20,
            },
        ]
    )

    lookup = build_row_lookup(df)

    assert len(lookup) == 2

    key = ("1", "L", "CC", 2020)
    assert key in lookup
    assert lookup[key]["value"] == 10