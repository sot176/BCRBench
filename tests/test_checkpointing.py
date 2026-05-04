"""
Tests for checkpointing and model state management.

Tests checkpoint creation, loading, validation, and recovery.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.checkpointing
class TestCheckpointCreation:
    """Test checkpoint creation."""
    
    def test_create_simple_checkpoint(self, temp_dir):
        """Test creating a simple checkpoint."""
        model = nn.Linear(10, 5)
        path = temp_dir / "simple.pth"
        
        torch.save(model.state_dict(), path)
        
        assert path.exists()
        assert path.stat().st_size > 0
    
    def test_create_checkpoint_with_metadata(self, temp_dir):
        """Test checkpoint with training metadata."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.234,
            'metrics': {
                'accuracy': 0.95,
                'c_index': 0.85
            }
        }
        
        path = temp_dir / "checkpoint_meta.pth"
        torch.save(checkpoint, path)
        
        loaded = torch.load(path, weights_only=False)
        assert loaded['epoch'] == 10
        assert loaded['loss'] == 0.234
        assert 'accuracy' in loaded['metrics']
    
    def test_checkpoint_file_size(self, temp_dir):
        """Test checkpoint file size is reasonable."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        path = temp_dir / "size_test.pth"
        torch.save(model.state_dict(), path)
        
        file_size = path.stat().st_size
        assert file_size > 1000  # At least 1KB
        assert file_size < 10_000_000  # Less than 10MB


@pytest.mark.checkpointing
class TestCheckpointLoading:
    """Test checkpoint loading."""
    
    def test_load_simple_checkpoint(self, temp_dir):
        """Test loading a simple checkpoint."""
        model = nn.Linear(10, 5)
        original_weight = model.weight.data.clone()
        
        # Save
        path = temp_dir / "load_test.pth"
        torch.save(model.state_dict(), path)
        
        # Modify model
        model.weight.data.fill_(0)
        assert not torch.allclose(model.weight, original_weight)
        
        # Load
        model.load_state_dict(torch.load(path))
        assert torch.allclose(model.weight, original_weight)
    
    def test_load_with_weight_only(self, temp_dir):
        """Test loading with weights_only=True."""
        model = nn.Linear(10, 5)
        path = temp_dir / "weights.pth"
        torch.save(model.state_dict(), path)
        
        loaded = torch.load(path, weights_only=True)
        assert 'weight' in loaded
        assert 'bias' in loaded
    
    def test_load_missing_checkpoint(self):
        """Test error handling for missing checkpoint."""
        with pytest.raises(FileNotFoundError):
            torch.load("/nonexistent/path/checkpoint.pth")
    
    def test_load_corrupted_checkpoint(self, temp_dir):
        """Test error handling for corrupted checkpoint."""
        path = temp_dir / "corrupted.pth"
        path.write_text("This is not a valid checkpoint file")
        
        with pytest.raises(Exception):
            torch.load(path)


@pytest.mark.checkpointing
class TestCheckpointValidation:
    """Test checkpoint validation."""
    
    def test_validate_state_dict_keys(self, temp_dir):
        """Test that loaded state dict has correct keys."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        
        path = temp_dir / "validate.pth"
        torch.save(model.state_dict(), path)
        
        loaded = torch.load(path)
        
        # Check expected keys exist
        assert '0.weight' in loaded
        assert '0.bias' in loaded
        assert '1.weight' in loaded
        assert '1.bias' in loaded
    
    def test_validate_tensor_shapes(self, temp_dir):
        """Test that loaded tensors have correct shapes."""
        model = nn.Linear(10, 5)
        path = temp_dir / "shapes.pth"
        torch.save(model.state_dict(), path)
        
        loaded = torch.load(path)
        
        assert loaded['weight'].shape == (5, 10)
        assert loaded['bias'].shape == (5,)
    
    def test_validate_checkpoint_compatibility(self, temp_dir):
        """Test checkpoint compatibility with model."""
        model1 = nn.Linear(10, 5)
        path = temp_dir / "compat.pth"
        torch.save(model1.state_dict(), path)
        
        model2 = nn.Linear(10, 5)
        checkpoint = torch.load(path)
        
        # Should load without error
        model2.load_state_dict(checkpoint)
        assert True
    
    def test_checkpoint_incompatible_model(self, temp_dir):
        """Test error when checkpoint doesn't match model."""
        model1 = nn.Linear(10, 5)
        path = temp_dir / "incompat.pth"
        torch.save(model1.state_dict(), path)
        
        model2 = nn.Linear(20, 10)  # Different size
        checkpoint = torch.load(path)
        
        with pytest.raises(RuntimeError):
            model2.load_state_dict(checkpoint)


@pytest.mark.checkpointing
class TestCheckpointRecovery:
    """Test model recovery from checkpoints."""
    
    def test_resume_training_epoch(self, temp_dir):
        """Test resuming training from epoch checkpoint."""
        initial_epoch = 5
        
        checkpoint = {
            'epoch': initial_epoch,
            'model_state': nn.Linear(10, 5).state_dict(),
            'optimizer_state': torch.optim.Adam(nn.Linear(10, 5).parameters()).state_dict(),
        }
        
        path = temp_dir / "resume_epoch.pth"
        torch.save(checkpoint, path)
        
        loaded = torch.load(path, weights_only=False)
        resume_epoch = loaded['epoch'] + 1
        
        assert resume_epoch == initial_epoch + 1
    
    def test_recover_best_model(self, temp_dir):
        """Test recovering best model from checkpoint."""
        model = nn.Linear(10, 5)
        best_path = temp_dir / "best.pth"
        
        # Simulate training
        best_loss = float('inf')
        for epoch in range(5):
            loss = 1.0 / (epoch + 1)
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), best_path)
        
        # Recover
        model_recovered = nn.Linear(10, 5)
        model_recovered.load_state_dict(torch.load(best_path))
        
        assert best_path.exists()
    
    def test_checkpoint_gives_exact_state(self, temp_dir):
        """Test that checkpoint restores exact model state."""
        model1 = nn.Linear(10, 5)
        model1.weight.data.fill_(1.5)
        model1.bias.data.fill_(0.5)
        
        path = temp_dir / "exact_state.pth"
        torch.save(model1.state_dict(), path)
        
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(torch.load(path))
        
        assert torch.allclose(model1.weight, model2.weight)
        assert torch.allclose(model1.bias, model2.bias)


@pytest.mark.checkpointing
class TestCheckpointCleanup:
    """Test checkpoint cleanup and management."""
    
    def test_checkpoint_directory_structure(self, temp_dir):
        """Test organizing checkpoints in directories."""
        run_dir = temp_dir / "run_001"
        run_dir.mkdir()
        
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        
        model = nn.Linear(10, 5)
        torch.save(model.state_dict(), checkpoint_dir / "epoch_01.pth")
        torch.save(model.state_dict(), checkpoint_dir / "epoch_02.pth")
        torch.save(model.state_dict(), checkpoint_dir / "best.pth")
        
        files = list(checkpoint_dir.glob("*.pth"))
        assert len(files) == 3
    
    def test_keep_best_checkpoint_only(self, temp_dir):
        """Test keeping only best checkpoint."""
        checkpoints_dir = temp_dir / "checkpoints"
        checkpoints_dir.mkdir()
        
        model = nn.Linear(10, 5)
        
        # Create multiple checkpoint files
        for i in range(5):
            checkpoint_file = checkpoints_dir / f"checkpoint_{i}.pth"
            torch.save(model.state_dict(), checkpoint_file)
        
        # Keep only best
        best_file = checkpoints_dir / "best.pth"
        torch.save(model.state_dict(), best_file)
        
        # In real scenario, delete old checkpoints
        for f in checkpoints_dir.glob("checkpoint_*.pth"):
            f.unlink()
        
        remaining = list(checkpoints_dir.glob("*.pth"))
        assert len(remaining) == 1
        assert remaining[0].name == "best.pth"


@pytest.mark.checkpointing
class TestCheckpointDifferentFormats:
    """Test different checkpoint formats."""
    
    def test_checkpoint_state_dict_only(self, temp_dir):
        """Test checkpoint with state dict only."""
        model = nn.Linear(10, 5)
        path = temp_dir / "state_dict.pth"
        torch.save(model.state_dict(), path)
        
        loaded = torch.load(path)
        new_model = nn.Linear(10, 5)
        new_model.load_state_dict(loaded)
        
        assert isinstance(loaded, dict)
    
    def test_checkpoint_entire_model(self, temp_dir):
        """Test checkpoint with entire model."""
        model = nn.Linear(10, 5)
        path = temp_dir / "model.pth"
        torch.save(model, path)
        
        loaded = torch.load(path, weights_only=False)
        assert isinstance(loaded, nn.Linear)
    
    def test_checkpoint_with_config(self, temp_dir):
        """Test checkpoint with model config."""
        model_config = {
            'input_dim': 10,
            'hidden_dim': 20,
            'output_dim': 5,
            'dropout': 0.1
        }
        
        model = nn.Linear(model_config['input_dim'], model_config['output_dim'])
        
        checkpoint = {
            'config': model_config,
            'model_state': model.state_dict(),
            'epoch': 10
        }
        
        path = temp_dir / "with_config.pth"
        torch.save(checkpoint, path)
        
        loaded = torch.load(path, weights_only=False)
        assert loaded['config']['input_dim'] == 10
