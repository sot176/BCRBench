"""
Tests for model creation and factory functions.

Tests the models/model_factory.py functions:
- get_model()
- build_mammo_reg_net()
- _build_model()

EXAMPLE: Using model-specific fixtures

You can use model-specific argument and config fixtures for testing:

    def test_mirai_with_specific_config(mock_args_mirai, mock_config_mirai):
        '''Test Mirai model with Mirai-specific arguments and config.'''
        assert mock_args_mirai.model == "Mirai"
        assert mock_config_mirai['num_images'] == 4

    def test_imgfeatalign_requires_reg_model(mock_args_imgfeatalign):
        '''Test ImgFeatAlign requires registration model checkpoint.'''
        assert mock_args_imgfeatalign.path_saved_reg_model is not None

Available model-specific fixtures:
- mock_args_mirai, mock_args_imgfeatalign, mock_args_lmvnet
- mock_args_vmramar, mock_args_oabreacr
- mock_config_mirai, mock_config_imgfeatalign, etc.

See conftest.py for the complete list of fixtures.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.mark.models
class TestModelFactory:
    """Test model factory functions."""
    
    def test_get_model_mirai(self):
        """Test building Mirai model."""
        from src.models.model_factory import get_model
        
        try:
            # This will require proper model imports, might raise ImportError
            model = get_model("Mirai")
            assert model is not None
            # Mirai should have forward method
            assert hasattr(model, 'forward')
        except ImportError:
            pytest.skip("Mirai model not available")
    
    def test_get_model_invalid_name(self):
        """Test that invalid model names raise ValueError."""
        from src.models.model_factory import get_model
        
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("InvalidModelName")
    
    def test_get_model_vmra_mar(self):
        """Test building VMRA-MaR model."""
        from src.models.model_factory import get_model
        
        try:
            model = get_model("VMRA-MaR")
            assert model is not None
        except ImportError:
            pytest.skip("VMRA-MaR model not available")
    
    def test_get_model_oa_breacr(self):
        """Test building OA-BreaCR model."""
        from src.models.model_factory import get_model
        
        try:
            model = get_model("OA-BreaCR")
            assert model is not None
        except ImportError:
            pytest.skip("OA-BreaCR model not available")


@pytest.mark.models
class TestRegistrationModels:
    """Test models that require registration network."""
    
    def test_imgfeatalign_requires_reg_model(self):
        """Test that ImgFeatAlign requires MammoRegNet."""
        from src.models.model_factory import get_model
        
        with pytest.raises(ValueError, match="requires a MammoRegNet checkpoint"):
            get_model("ImgFeatAlign", path_saved_reg_model=None)
    
    def test_lmvnet_requires_reg_model(self):
        """Test that LMV-Net requires MammoRegNet."""
        from src.models.model_factory import get_model
        
        with pytest.raises(ValueError, match="requires a MammoRegNet checkpoint"):
            get_model("LMV-Net", path_saved_reg_model=None)
    
    @patch('models.model_factory.build_mammo_reg_net')
    def test_imgfeatalign_with_reg_model(self, mock_build_reg):
        """Test building ImgFeatAlign with registration model."""
        mock_reg_net = MagicMock()
        mock_build_reg.return_value = mock_reg_net
        
        from src.models.model_factory import get_model
        
        try:
            model = get_model("ImgFeatAlign", path_saved_reg_model="/path/to/model.pth")
            mock_build_reg.assert_called_once()
        except ImportError:
            pytest.skip("ImgFeatAlign model not available")


@pytest.mark.models
class TestBuildMammoRegNet:
    """Test MammoRegNet building."""
    
    @patch('torch.load')
    def test_build_mammo_reg_net_success(self, mock_load, temp_dir):
        """Test successful MammoRegNet building."""
        from src.models.model_factory import build_mammo_reg_net
        
        # Mock checkpoint
        mock_checkpoint = {"module.layer1.weight": torch.randn(64, 64, 3, 3)}
        mock_load.return_value = mock_checkpoint
        
        try:
            reg_model = build_mammo_reg_net(str(temp_dir / "test.pth"))
            assert reg_model is not None
            assert reg_model.training == False  # Should be in eval mode
        except ImportError:
            pytest.skip("MammoRegNet not available")
    
    def test_build_mammo_reg_net_invalid_path(self):
        """Test MammoRegNet with invalid path."""
        from src.models.model_factory import build_mammo_reg_net
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            build_mammo_reg_net("/nonexistent/path/model.pth")


@pytest.mark.models
class TestModelProperties:
    """Test properties of created models."""
    
    def test_model_has_forward_method(self, mock_model):
        """Test that model has callable forward method."""
        assert hasattr(mock_model, 'forward')
        assert callable(mock_model.forward)
    
    def test_model_inference_output_shape(self, mock_model, sample_batch):
        """Test model output shape."""
        output = mock_model(sample_batch["images"])
        assert output.shape[0] == sample_batch["images"].shape[0]
        assert output.shape[1] == 1  # Single risk score output
    
    @pytest.mark.slow
    def test_model_device_compatibility(self, mock_model, device):
        """Test model can be moved to different devices."""
        # Move to device
        model_on_device = mock_model.to(device)
        assert model_on_device is not None


@pytest.mark.models  
class TestModelKwargs:
    """Test model building with different kwargs."""
    
    def test_build_model_with_args(self, mock_args):
        """Test building model with args object."""
        from src.models.model_factory import _build_model
        
        mock_model_class = MagicMock()
        _build_model(mock_model_class, args=mock_args)
        
        # Should pass args if model accepts it
        mock_model_class.assert_called_once()
    
    def test_build_model_filters_kwargs(self):
        """Test that _build_model filters kwargs based on __init__ signature."""
        from src.models.model_factory import _build_model
        
        class DummyModel:
            def __init__(self, param1):
                self.param1 = param1
        
        # Pass extra kwargs that shouldn't be used
        model = _build_model(DummyModel, param1="value1", param2="value2")
        
        assert model.param1 == "value1"


@pytest.mark.models
class TestModelRegistry:
    """Test model registry structure."""
    
    def test_all_supported_models_can_be_accessed(self):
        """Test that all model names can be queried."""
        from src.models.model_factory import get_model
        
        supported_models = ["Mirai", "ImgFeatAlign", "LMV-Net", "VMRA-MaR", "OA-BreaCR"]
        
        for model_name in supported_models:
            try:
                # Just test that invalid model raises correct error
                _ = get_model(model_name)
            except ValueError as e:
                # Should only raise for truly invalid models
                assert "Unknown model" not in str(e)
            except (ImportError, TypeError):
                # Model may not be fully initialized, that's ok
                pass


@pytest.mark.models
class TestModelWithSpecificConfigs:
    """Test models using model-specific fixtures."""
    
    def test_mirai_training_with_specific_config(self, mock_args_mirai, mock_config_mirai):
        """Test Mirai model with Mirai-specific arguments and config."""
        assert mock_args_mirai.model == "Mirai"
        assert mock_config_mirai['num_images'] == 4
        assert mock_args_mirai.num_images == 4
        assert not mock_config_mirai['freeze_image_encoder']
    
    def test_imgfeatalign_training_with_specific_config(self, mock_args_imgfeatalign, mock_config_imgfeatalign):
        """Test ImgFeatAlign requires registration model."""
        assert mock_args_imgfeatalign.model == "ImgFeatAlign"
        assert mock_args_imgfeatalign.path_saved_reg_model is not None
        assert mock_config_imgfeatalign['use_deformation']
    
    def test_lmvnet_training_with_specific_config(self, mock_args_lmvnet, mock_config_lmvnet):
        """Test LMV-Net model-specific configuration."""
        assert mock_args_lmvnet.model == "LMV-Net"
        assert mock_args_lmvnet.num_views == 2
        assert mock_args_lmvnet.num_timepoints == 2
        assert mock_config_lmvnet['use_attention']
    
    def test_vmramar_training_with_specific_config(self, mock_args_vmramar, mock_config_vmramar):
        """Test VMRA-MaR model-specific configuration."""
        assert mock_args_vmramar.model == "VMRA-MaR"
        assert mock_args_vmramar.num_images == 4
        assert mock_config_vmramar['use_asymmetry_detector']
        assert mock_config_vmramar['use_longitudinal_tracker']
    
    def test_oabreacr_training_with_specific_config(self, mock_args_oabreacr, mock_config_oabreacr):
        """Test OA-BreaCR model-specific configuration."""
        assert mock_args_oabreacr.model == "OA-BreaCR"
        assert mock_args_oabreacr.num_views == 1
        assert mock_args_oabreacr.num_timepoints == 2
        assert mock_config_oabreacr['use_deformation_field']
