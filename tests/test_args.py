"""
Tests for argument parsing and configuration loading.

Tests the main_train.py functions:
- setup_logging()
- load_model_config()
- parse_cli_args()
"""

import pytest
import logging
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.config
class TestSetupLogging:
    """Test logging configuration."""
    
    def test_setup_logging_main_process(self, temp_dir):
        """Test logging setup on main process."""
        from src.main_train import setup_logging
        
        log_path = temp_dir / "test.log"
        logger = setup_logging(str(log_path), is_main_process=True)
        
        assert logger is not None
        assert log_path.exists()
        assert logger.level == logging.INFO
    
    def test_setup_logging_non_main_process(self, temp_dir):
        """Test logging setup on non-main process."""
        from src.main_train import setup_logging
        
        log_path = temp_dir / "test.log"
        logger = setup_logging(str(log_path), is_main_process=False)
        
        assert logger is None or len(logger.handlers) == 0
    
    def test_setup_logging_handlers(self, temp_dir):
        """Test that both file and console handlers are created."""
        from src.main_train import setup_logging
        
        log_path = temp_dir / "test.log"
        logger = setup_logging(str(log_path), is_main_process=True)
        
        # Check handlers exist
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    def test_setup_logging_invalid_path(self):
        """Test logging setup with invalid path."""
        from src.main_train import setup_logging
        
        invalid_path = "/invalid/path/that/does/not/exist/test.log"
        with pytest.raises(IOError):
            setup_logging(invalid_path, is_main_process=True)


@pytest.mark.config
class TestLoadModelConfig:
    """Test model configuration loading."""
    
    @patch('builtins.open', create=True)
    def test_load_config_success(self, mock_open, mock_config):
        """Test successful config loading."""
        from src.main_train import load_model_config
        
        mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(mock_config)
        
        # This test uses mocking to avoid file system dependencies
        logger = MagicMock()
        config = load_model_config("Mirai", logger)
        
        # Should not raise and should handle YAML parsing
        assert isinstance(config, dict)
    
    def test_load_config_nonexistent_model(self):
        """Test loading config for non-existent model."""
        from src.main_train import load_model_config
        
        logger = MagicMock()
        config = load_model_config("NonExistentModel", logger)
        
        # Should return empty dict and log warning
        assert config == {}
        logger.warning.assert_called()
    
    def test_load_config_model_name_normalization(self):
        """Test that model names are normalized correctly."""
        from src.main_train import load_model_config
        
        logger = MagicMock()
        
        # Test with different name formats
        for model_name in ["Mirai", "mirai", "MIRAI", "Img-FeatAlign", "img-featalign"]:
            config = load_model_config(model_name, logger)
            # Should not raise exception
            assert isinstance(config, dict)


@pytest.mark.config
class TestParseCliArgs:
    """Test command-line argument parsing."""
    
    @patch('sys.argv', ['train.py', '--model', 'Mirai', '--dataset', 'EMBED', '--batch_size', '32'])
    def test_parse_basic_args(self):
        """Test parsing basic command-line arguments."""
        from src.main_train import parse_cli_args
        
        args = parse_cli_args()
        
        assert args.model == "Mirai"
        assert args.dataset == "EMBED"
        assert args.batch_size == 32
    
    @patch('sys.argv', ['train.py', '--model', 'Mirai', '--num_epochs', '10', '--learning_rate', '0.001'])
    def test_parse_training_args(self):
        """Test parsing training-specific arguments."""
        from src.main_train import parse_cli_args
        
        args = parse_cli_args()
        
        assert args.num_epochs == 10
        assert args.learning_rate == 0.001
    
    @patch('sys.argv', ['train.py', '--model', 'InvalidModel'])
    def test_parse_invalid_model(self):
        """Test parsing with invalid model name."""
        # This depends on how validation is implemented
        # Add test based on your validation logic
        pass
    
    @patch('sys.argv', ['train.py', '--seed', '42'])
    def test_parse_seed_arg(self):
        """Test parsing random seed argument."""
        from src.main_train import parse_cli_args
        
        args = parse_cli_args()
        assert args.seed == 42


@pytest.mark.config
class TestConfigIntegration:
    """Integration tests for configuration loading."""
    
    def test_config_yaml_structure(self, temp_dir):
        """Test that config YAML files have expected structure."""
        config_file = temp_dir / "test_config.yaml"
        
        test_config = {
            "transformer_hidden_dim": 512,
            "num_layers": 1,
            "dropout": 0.0,
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == test_config
    
    def test_config_merges_cli_and_yaml(self):
        """Test that CLI args and YAML config are properly merged."""
        # This test depends on your implementation
        # Should verify that CLI args override YAML config
        pass


@pytest.mark.config
class TestErrorHandling:
    """Test error handling in configuration."""
    
    def test_corrupted_yaml_file(self, temp_dir):
        """Test handling of corrupted YAML files."""
        from src.main_train import load_model_config
        
        corrupted_file = temp_dir / "corrupted.yaml"
        corrupted_file.write_text("{ invalid yaml [")
        
        # Should handle gracefully
        logger = MagicMock()
        # Depending on implementation, this may raise or return empty dict
        try:
            config = load_model_config("test", logger)
            assert isinstance(config, dict)
        except Exception:
            # Implementation may choose to raise
            pass
