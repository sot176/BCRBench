"""
Pytest configuration and shared fixtures for BreastCancerRiskBenchmark tests.

This module provides:
- Pytest configuration
- Shared fixtures for models, datasets, and test data
- Mock objects for external dependencies

AVAILABLE FIXTURES:

Training Arguments (mock_args):
  - mock_args               : Default Mirai model args
  - mock_args_mirai         : Mirai-specific args
  - mock_args_imgfeatalign  : ImgFeatAlign-specific args
  - mock_args_lmvnet        : LMV-Net-specific args
  - mock_args_vmramar       : VMRA-MaR-specific args
  - mock_args_oabreacr      : OA-BreaCR-specific args

Model Configurations (mock_config):
  - mock_config             : Default configuration
  - mock_config_mirai       : Mirai-specific config
  - mock_config_imgfeatalign: ImgFeatAlign-specific config
  - mock_config_lmvnet      : LMV-Net-specific config
  - mock_config_vmramar     : VMRA-MaR-specific config
  - mock_config_oabreacr    : OA-BreaCR-specific config

Mock Objects:
  - mock_model              : Mock PyTorch model
  - mock_dataloader         : Mock dataloader

Test Data:
  - sample_predictions      : Random predictions array (100 samples)
  - sample_event_times      : Sample survival times (100 samples)
  - sample_event_observed   : Event indicators (100 samples)
  - sample_batch            : Sample image batch (4, 3, 224, 224)
  - censoring_dist          : Sample censoring distribution dict

Utilities:
  - temp_dir                : Temporary directory for test files
  - device                  : Appropriate device (CPU or CUDA)
  - project_root            : Project root path

EXAMPLE USAGE:

  def test_mirai_training(mock_args_mirai, mock_config_mirai):
      '''Test Mirai model with model-specific config.'''
      assert mock_args_mirai.model == "Mirai"
      assert mock_config_mirai['num_images'] == 4

  def test_imgfeatalign_training(mock_args_imgfeatalign):
      '''Test ImgFeatAlign with its specific args.'''
      assert mock_args_imgfeatalign.path_saved_reg_model is not None
"""

from types import SimpleNamespace

import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator


# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import pytest

@pytest.fixture
def accelerator():
    return Accelerator(cpu=True)

@pytest.fixture
def args():
    return SimpleNamespace(
        lr=1e-3,
        epochs=1,
        batch_size=4,
        device="cpu",
        model="ImgFeatAlign"
    )

@pytest.fixture(autouse=True)
def cleanup_logging():
    yield

    # clear your app logger
    logger = logging.getLogger("main_train")
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)

    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_args():
    """Create mock training arguments (default for Mirai)."""
    args = MagicMock()
    args.model = "Mirai"
    args.dataset = "EMBED"
    args.batch_size = 4
    args.num_epochs = 2
    args.learning_rate = 1e-4
    args.device = "cpu"
    args.num_workers = 0
    args.seed = 42
    args.save_dir = "/tmp/models"
    args.log_dir = "/tmp/logs"
    return args


@pytest.fixture
def mock_args_mirai():
    """Create mock training arguments for Mirai model."""
    args = MagicMock()
    args.model = "Mirai"
    args.dataset = "EMBED"
    args.batch_size = 4
    args.num_epochs = 2
    args.learning_rate = 1e-4
    args.device = "cpu"
    args.num_workers = 0
    args.seed = 42
    args.save_dir = "/tmp/models"
    args.log_dir = "/tmp/logs"
    args.num_images = 4
    args.freeze_image_encoder = False
    return args


@pytest.fixture
def mock_args_imgfeatalign():
    """Create mock training arguments for ImgFeatAlign model."""
    args = MagicMock()
    args.model = "ImgFeatAlign"
    args.dataset = "EMBED"
    args.batch_size = 4
    args.num_epochs = 2
    args.learning_rate = 5e-5
    args.device = "cpu"
    args.num_workers = 0
    args.seed = 42
    args.save_dir = "/tmp/models"
    args.log_dir = "/tmp/logs"
    args.path_saved_reg_model = "/path/to/reg_model.pth"
    return args


@pytest.fixture
def mock_args_lmvnet():
    """Create mock training arguments for LMV-Net model."""
    args = MagicMock()
    args.model = "LMV-Net"
    args.dataset = "EMBED"
    args.batch_size = 4
    args.num_epochs = 2
    args.learning_rate = 5e-5
    args.device = "cpu"
    args.num_workers = 0
    args.seed = 42
    args.save_dir = "/tmp/models"
    args.log_dir = "/tmp/logs"
    args.num_views = 2
    args.num_timepoints = 2
    args.path_saved_reg_model = "/path/to/reg_model.pth"
    return args


@pytest.fixture
def mock_args_vmramar():
    """Create mock training arguments for VMRA-MaR model."""
    args = MagicMock()
    args.model = "VMRA-MaR"
    args.dataset = "EMBED"
    args.batch_size = 4
    args.num_epochs = 2
    args.learning_rate = 1e-4
    args.device = "cpu"
    args.num_workers = 0
    args.seed = 42
    args.save_dir = "/tmp/models"
    args.log_dir = "/tmp/logs"
    args.num_images = 4
    return args


@pytest.fixture
def mock_args_oabreacr():
    """Create mock training arguments for OA-BreaCR model."""
    args = MagicMock()
    args.model = "OA-BreaCR"
    args.dataset = "EMBED"
    args.batch_size = 4
    args.num_epochs = 2
    args.learning_rate = 5e-5
    args.device = "cpu"
    args.num_workers = 0
    args.seed = 42
    args.save_dir = "/tmp/models"
    args.log_dir = "/tmp/logs"
    args.num_views = 1
    args.num_timepoints = 2
    return args


@pytest.fixture
def mock_config():
    """Create mock model configuration (default)."""
    return {
        "transformer_hidden_dim": 512,
        "num_layers": 1,
        "num_heads": 8,
        "dropout": 0.0,
        "num_images": 4,
        "survival_analysis_setup": True,
        "max_followup": 5,
    }


@pytest.fixture
def mock_config_mirai():
    """Create mock Mirai model configuration."""
    return {
        "model_name": "mirai_full",
        "transformer_hidden_dim": 512,
        "num_layers": 1,
        "num_heads": 8,
        "dropout": 0.0,
        "num_chan": 3,
        "multi_image": True,
        "num_images": 4,
        "survival_analysis_setup": True,
        "max_followup": 5,
        "freeze_image_encoder": False,
    }


@pytest.fixture
def mock_config_imgfeatalign():
    """Create mock ImgFeatAlign model configuration."""
    return {
        "model_name": "imgfeatalign",
        "transformer_hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.1,
        "num_images": 2,
        "use_deformation": True,
        "survival_analysis_setup": True,
        "max_followup": 5,
    }


@pytest.fixture
def mock_config_lmvnet():
    """Create mock LMV-Net model configuration."""
    return {
        "model_name": "lmvnet",
        "transformer_hidden_dim": 512,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.0,
        "num_views": 2,
        "num_timepoints": 2,
        "use_attention": True,
        "survival_analysis_setup": True,
        "max_followup": 5,
    }


@pytest.fixture
def mock_config_vmramar():
    """Create mock VMRA-MaR model configuration."""
    return {
        "model_name": "vmramar",
        "transformer_hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.0,
        "num_images": 4,
        "num_timepoints": 4,
        "use_asymmetry_detector": True,
        "use_longitudinal_tracker": True,
        "survival_analysis_setup": True,
        "max_followup": 5,
    }


@pytest.fixture
def mock_config_oabreacr():
    """Create mock OA-BreaCR model configuration."""
    return {
        "model_name": "oabreacr",
        "transformer_hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.1,
        "num_views": 1,
        "num_timepoints": 2,
        "use_deformation_field": True,
        "survival_analysis_setup": True,
        "max_followup": 5,
    }


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing evaluation metrics."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def sample_event_times():
    """Create sample event times for testing C-index."""
    np.random.seed(42)
    return np.random.rand(100) * 5  # 0-5 years


@pytest.fixture
def sample_event_observed():
    """Create sample event indicators."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images for model testing."""
    batch = {
        "images": torch.randn(4, 3, 224, 224),  # batch_size=4, 3 channels, 224x224
        "labels": torch.randint(0, 2, (4,)),
        "patient_id": ["patient_1", "patient_2", "patient_3", "patient_4"],
    }
    return batch


@pytest.fixture
def mock_dataloader():

    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 2, (8,)).float()

    dataset = TensorDataset(images, labels)

    class WrappedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, idx):
            image, label = dataset[idx]

            return {
                "images": image,
                "labels": label,
                "event_times": torch.tensor(1.0),
                "event_observed": torch.tensor(1),
            }

    return DataLoader(WrappedDataset(), batch_size=4)


@pytest.fixture
def mock_model():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 224 * 224, 1)
            )

        def forward(self, batch):
            if isinstance(batch, dict):
                x = batch["images"]
            else:
                x = batch

            return self.net(x)

        def get_primary_risk_head(self, outputs):
            return outputs

    return TinyModel()


@pytest.fixture
def censoring_dist():
    """Create sample censoring distribution."""
    return {
        0.0: 0.99,
        1.0: 0.95,
        2.0: 0.90,
        3.0: 0.85,
        4.0: 0.80,
        5.0: 0.75,
    }


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


# Session-scoped fixtures for expensive operations
@pytest.fixture(scope="session")
def device():
    """Get the appropriate device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch():
    """Create a sample batch for training/evaluation tests."""
    labels = torch.randint(0, 2, (4,))
    return {
        "images": torch.randn(4, 3, 224, 224),
        "labels": labels,
        "event_times": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "event_observed": labels.clone().long(),
        "patient_id": ["patient_1", "patient_2", "patient_3", "patient_4"],
    }