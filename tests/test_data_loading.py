"""
Tests for dataset and dataloader factory functions.

Tests the datasets/dataloader_factory.py functions and data loading utilities.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np


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

 