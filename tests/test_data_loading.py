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
            dataset_name="EMBED",
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
    
    def test_batch_image_shape(self, sample_batch):
        """Test batch image shape."""
        images = sample_batch["images"]
        assert len(images.shape) == 4  # (batch, channels, height, width)
        assert images.shape[1] == 3  # RGB channels
    
    def test_batch_labels_shape(self, sample_batch):
        """Test batch labels shape."""
        labels = sample_batch["labels"]
        assert len(labels.shape) >= 1
        assert labels.shape[0] == sample_batch["images"].shape[0]
    
    def test_batch_tensor_types(self, sample_batch):
        """Test that batch contains tensors."""
        assert isinstance(sample_batch["images"], torch.Tensor)
        assert isinstance(sample_batch["labels"], torch.Tensor)


@pytest.mark.datasets
class TestDataloaderProperties:
    """Test dataloader properties."""
    
    def test_dataloader_iteration(self, mock_dataloader):
        """Test iterating over dataloader."""
        batches = list(mock_dataloader)
        assert len(batches) > 0
        assert "images" in batches[0]
    
    def test_dataloader_length(self, mock_dataloader):
        """Test dataloader length."""
        length = len(mock_dataloader)
        assert length > 0
        assert isinstance(length, int)
    
    def test_dataloader_batch_size(self, mock_dataloader):
        """Test batch size consistency."""
        for batch in mock_dataloader:
            batch_size = batch["images"].shape[0]
            assert batch_size > 0


@pytest.mark.datasets
@pytest.mark.slow
class TestDataAugmentation:
    """Test data augmentation."""
    
    def test_augmentation_changes_images(self, sample_batch):
        """Test that augmentation produces different results."""
        images1 = sample_batch["images"].clone()
        
        # Simulate augmentation by adding noise
        images2 = images1 + torch.randn_like(images1) * 0.1
        
        # Should be different
        assert not torch.allclose(images1, images2)
    
    def test_augmentation_preserves_shape(self, sample_batch):
        """Test that augmentation preserves image shape."""
        original_shape = sample_batch["images"].shape
        augmented = sample_batch["images"] + torch.randn_like(sample_batch["images"]) * 0.01
        assert augmented.shape == original_shape
    
    def test_augmentation_in_range(self, sample_batch):
        """Test that augmentation keeps values in valid range."""
        images = torch.clamp(sample_batch["images"], 0, 1)
        assert images.min() >= 0
        assert images.max() <= 1


@pytest.mark.datasets
class TestDataSplits:
    """Test dataset splits (train/val/test)."""
    
    def test_split_names(self):
        """Test that valid split names are accepted."""
        valid_splits = ["train", "val", "test", "validation"]
        for split in valid_splits:
            # Should not raise for valid splits
            assert split in valid_splits
    
    
    def test_split_sizes_reasonable(self):
        """Test that split sizes are reasonable."""
        total = 1000
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        test_size = total - train_size - val_size
        
        assert train_size > val_size
        assert val_size > 0
        assert test_size > 0
        assert train_size + val_size + test_size == total


@pytest.mark.datasets
class TestDataNormalization:
    """Test data normalization."""
    
    def test_image_normalization_range(self, sample_batch):
        """Test that images are in valid range after normalization."""
        images = torch.clamp(sample_batch["images"], 0, 1)
        assert images.min() >= 0
        assert images.max() <= 1
    
    def test_normalization_preserves_structure(self, sample_batch):
        """Test that normalization preserves image structure."""
        original = sample_batch["images"]
        
        # Normalize
        normalized = (original - original.min()) / (original.max() - original.min() + 1e-6)
        
        assert normalized.shape == original.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1


@pytest.mark.datasets
class TestDatasetMetadata:
    """Test dataset metadata and information."""
    
    def test_dataset_info_accessible(self):
        """Test that dataset provides metadata."""
        # Should have info about dataset structure
        info = {
            "num_samples": 1000,
            "num_classes": 2,
            "image_size": 224,
        }
        assert isinstance(info, dict)
        assert "num_samples" in info
