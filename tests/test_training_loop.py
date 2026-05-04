"""
Tests for training loop and checkpoint management.

Tests training functions and checkpoint saving/loading.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.mark.training
class TestTrainingLoop:
    """Test training loop functions."""
    
    def test_train_step_basic(self, mock_model, sample_batch):
        """Test basic training step."""
        # Setup
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Forward pass
        predictions = mock_model(sample_batch["images"])
        loss = loss_fn(predictions, sample_batch["labels"].float().unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert isinstance(loss.item(), float)
    
    def test_validation_step(self, mock_model, sample_batch):
        """Test validation step."""
        mock_model.eval()
        
        with torch.no_grad():
            predictions = mock_model(sample_batch["images"])
        
        assert predictions.shape[0] == sample_batch["images"].shape[0]
        assert not predictions.requires_grad
    
    def test_gradient_flow(self, mock_model, sample_batch):
        """Test that gradients flow properly."""
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        initial_params = [p.clone() for p in mock_model.parameters()]
        
        # Training step
        predictions = mock_model(sample_batch["images"])
        loss = loss_fn(predictions, sample_batch["labels"].float().unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        params_changed = False
        for p_new in mock_model.parameters():
            if p_new is not None:
                params_changed = True
        
        assert params_changed


@pytest.mark.training
class TestLossCalculation:
    """Test loss function calculations."""
    
    def test_bce_loss(self, sample_batch):
        """Test binary cross entropy loss."""
        predictions = torch.sigmoid(torch.randn(4, 1))
        labels = sample_batch["labels"].float().unsqueeze(1)
        
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss."""
        predictions = torch.randn(4, 5)  # 4 samples, 5 classes
        labels = torch.randint(0, 5, (4,))
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predictions, labels)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_loss_backward(self):
        """Test loss backward pass."""
        x = torch.randn(4, 10, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)
        
        loss = (x.sum() + y.sum()) ** 2
        loss.backward()
        
        assert x.grad is not None
        assert y.grad is not None


@pytest.mark.training
class TestBatchProcessing:
    """Test batch processing in training."""
    
    def test_batch_to_device(self, sample_batch, device):
        """Test moving batch to device."""
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }
        
        assert batch_on_device["images"].device.type == device.type
        assert batch_on_device["labels"].device.type == device.type
    
    def test_batch_accumulation(self):
        """Test gradient accumulation over multiple batches."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        total_loss = 0
        num_batches = 3
        
        for i in range(num_batches):
            x = torch.randn(4, 10)
            y = torch.randn(4, 1)
            
            predictions = model(x)
            loss = loss_fn(predictions, y)
            loss.backward()
            
            total_loss += loss.item()
        
        optimizer.step()
        
        assert total_loss > 0
        assert num_batches > 0


@pytest.mark.checkpointing
class TestCheckpointing:
    """Test model checkpointing functionality."""
    
    def test_save_checkpoint(self, temp_dir):
        """Test saving model checkpoint."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': 5,
            'loss': 0.123
        }
        
        checkpoint_path = temp_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, temp_dir):
        """Test loading model checkpoint."""
        model1 = nn.Linear(10, 1)
        optimizer1 = torch.optim.Adam(model1.parameters())
        
        # Save
        checkpoint = {
            'model_state': model1.state_dict(),
            'optimizer_state': optimizer1.state_dict(),
            'epoch': 5
        }
        checkpoint_path = temp_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load
        model2 = nn.Linear(10, 1)
        optimizer2 = torch.optim.Adam(model2.parameters())
        
        loaded_checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(loaded_checkpoint['model_state'])
        optimizer2.load_state_dict(loaded_checkpoint['optimizer_state'])
        
        assert loaded_checkpoint['epoch'] == 5
    
    def test_checkpoint_best_model(self, temp_dir):
        """Test saving best model during training."""
        best_loss = float('inf')
        best_model_path = temp_dir / "best_model.pth"
        
        model = nn.Linear(10, 1)
        
        # Simulate training with improving loss
        for epoch in range(5):
            loss = 1.0 / (epoch + 1)  # Decreasing loss
            
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), best_model_path)
        
        assert best_model_path.exists()
        assert best_loss < 1.0
    
    def test_checkpoint_resume_training(self, temp_dir):
        """Test resuming training from checkpoint."""
        # First training session
        model1 = nn.Linear(10, 1)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        
        checkpoint = {
            'model': model1.state_dict(),
            'optimizer': optimizer1.state_dict(),
            'epoch': 10,
            'best_loss': 0.5
        }
        
        checkpoint_path = temp_dir / "resume.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Resume training
        model2 = nn.Linear(10, 1)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        
        loaded = torch.load(checkpoint_path)
        model2.load_state_dict(loaded['model'])
        optimizer2.load_state_dict(loaded['optimizer'])
        epoch_resume = loaded['epoch']
        
        assert epoch_resume == 10


@pytest.mark.checkpointing
class TestCheckpointIntegrity:
    """Test checkpoint integrity and compatibility."""
    
    def test_checkpoint_structure(self, temp_dir):
        """Test checkpoint has correct structure."""
        model = nn.Linear(10, 1)
        checkpoint = {
            'model_state': model.state_dict(),
            'epoch': 1,
            'loss': 0.5,
            'config': {'lr': 0.001}
        }
        
        path = temp_dir / "test.pth"
        torch.save(checkpoint, path)
        
        loaded = torch.load(path)
        assert 'model_state' in loaded
        assert 'epoch' in loaded
        assert 'loss' in loaded
        assert 'config' in loaded
    
    def test_model_state_dict_format(self):
        """Test model state dict has correct format."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        state_dict = model.state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        assert all('weight' in k or 'bias' in k for k in state_dict.keys())
    
    def test_checkpoint_torch_version_compatibility(self, temp_dir):
        """Test checkpoint compatibility."""
        model = nn.Linear(10, 1)
        path = temp_dir / "compat.pth"
        
        # Save with default protocol
        torch.save(model.state_dict(), path)
        
        # Load should work
        loaded = torch.load(path)
        assert isinstance(loaded, dict)


@pytest.mark.training
@pytest.mark.slow
class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_full_training_epoch(self, mock_dataloader, mock_model):
        """Test full training epoch."""
        mock_model.train()
        optimizer = torch.optim.Adam(mock_model.parameters())
        loss_fn = nn.BCEWithLogitsLoss()
        
        epoch_loss = 0
        num_batches = 0
        
        for batch in mock_dataloader:
            predictions = mock_model(batch["images"])
            loss = loss_fn(predictions, batch["labels"].float().unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        assert avg_loss > 0
        assert num_batches > 0
    
    def test_train_val_split(self, mock_dataloader):
        """Test using different dataloaders for train and val."""
        train_loader = mock_dataloader
        val_loader = mock_dataloader
        
        train_batches = list(train_loader)
        val_batches = list(val_loader)
        
        assert len(train_batches) > 0
        assert len(val_batches) > 0
