"""
Tests for model creation and factory functions.

Tests the models/model_factory.py functions:
- get_model()
- build_mammo_reg_net()
- _build_model()
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

pytestmark = [
    pytest.mark.models,
    pytest.mark.filterwarnings(
        "ignore:Importing from timm.models.layers is deprecated.*:FutureWarning"
    ),
]


class TestModelFactory:
    def test_get_model_invalid_name(self):
        from src.models.model_factory import get_model

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("InvalidModelName")

    @patch("models.Mirai.model.Mirai")
    def test_get_model_mirai(self, mock_mirai, mock_args_mirai):
        from src.models.model_factory import get_model

        mock_instance = MagicMock()
        mock_mirai.return_value = mock_instance

        model = get_model("Mirai", args=mock_args_mirai)

        mock_mirai.assert_called_once()
        assert model is mock_instance

    @patch("models.VMRAMAR.model.VMRAMaR")
    def test_get_model_vmra_mar(self, mock_vmramar, mock_args_vmramar):
        from src.models.model_factory import get_model

        mock_instance = MagicMock()
        mock_vmramar.return_value = mock_instance

        model = get_model("VMRA-MaR", args=mock_args_vmramar)

        mock_vmramar.assert_called_once()
        assert model is mock_instance

    @patch("models.OABreaCR.model.OA_BreaCR")
    def test_get_model_oa_breacr(self, mock_oabreacr, mock_args_oabreacr):
        from src.models.model_factory import get_model

        mock_instance = MagicMock()
        mock_oabreacr.return_value = mock_instance

        model = get_model("OA-BreaCR", args=mock_args_oabreacr)

        mock_oabreacr.assert_called_once()
        assert model is mock_instance


class TestRegistrationModels:
    def test_imgfeatalign_requires_reg_model(self):
        from src.models.model_factory import get_model

        with pytest.raises(ValueError, match="requires a MammoRegNet checkpoint"):
            get_model("ImgFeatAlign", path_saved_reg_model=None)

    def test_lmvnet_requires_reg_model(self):
        from src.models.model_factory import get_model

        with pytest.raises(ValueError, match="requires a MammoRegNet checkpoint"):
            get_model("LMV-Net", path_saved_reg_model=None)

    @patch("models.ImgFeatAlign.model.ImgFeatAlign")
    @patch("src.models.model_factory.build_mammo_reg_net")
    def test_imgfeatalign_with_reg_model(self, mock_build_reg, mock_imgfeatalign, mock_args_imgfeatalign):
        from src.models.model_factory import get_model

        mock_reg_net = MagicMock()
        mock_build_reg.return_value = mock_reg_net

        mock_model = MagicMock()
        mock_imgfeatalign.return_value = mock_model

        model = get_model(
            "ImgFeatAlign",
            args=mock_args_imgfeatalign,
            path_saved_reg_model="/path/to/model.pth",
        )

        mock_build_reg.assert_called_once_with("/path/to/model.pth")
        mock_imgfeatalign.assert_called_once()
        assert model is mock_model

    @patch("models.LMVNet.model.LMVNet")
    @patch("src.models.model_factory.build_mammo_reg_net")
    def test_lmvnet_with_reg_model(self, mock_build_reg, mock_lmvnet, mock_args_lmvnet):
        from src.models.model_factory import get_model

        mock_reg_net = MagicMock()
        mock_build_reg.return_value = mock_reg_net

        mock_model = MagicMock()
        mock_lmvnet.return_value = mock_model

        model = get_model(
            "LMV-Net",
            args=mock_args_lmvnet,
            path_saved_reg_model="/path/to/model.pth",
        )

        mock_build_reg.assert_called_once_with("/path/to/model.pth")
        mock_lmvnet.assert_called_once()
        assert model is mock_model


class TestBuildMammoRegNet:
    @patch("src.models.model_factory.MammoRegNet")
    @patch("src.models.model_factory.torch.load")
    def test_build_mammo_reg_net_success(self, mock_load, mock_reg_class, temp_dir):
        from src.models.model_factory import build_mammo_reg_net

        mock_checkpoint = {"module.layer1.weight": torch.randn(1)}
        mock_load.return_value = mock_checkpoint

        mock_reg_instance = MagicMock()
        mock_reg_class.return_value = mock_reg_instance

        reg_model = build_mammo_reg_net(str(temp_dir / "test.pth"))

        mock_load.assert_called_once()
        mock_reg_class.assert_called_once()
        mock_reg_instance.load_state_dict.assert_called_once_with(
            {"layer1.weight": mock_checkpoint["module.layer1.weight"]}
        )
        mock_reg_instance.eval.assert_called_once()
        assert reg_model is mock_reg_instance

    def test_build_mammo_reg_net_invalid_path(self):
        from src.models.model_factory import build_mammo_reg_net

        with pytest.raises((FileNotFoundError, RuntimeError)):
            build_mammo_reg_net("/nonexistent/path/model.pth")


class TestModelProperties:
    def test_model_has_forward_method(self, mock_model):
        assert hasattr(mock_model, "forward")
        assert callable(mock_model.forward)

    def test_model_inference_output_shape(self, sample_batch):
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(sample_batch["images"].shape[0], 1)

        output = mock_model(sample_batch["images"])

        assert output.shape[0] == sample_batch["images"].shape[0]
        assert output.shape[1] == 1

    @pytest.mark.slow
    def test_model_device_compatibility(self, mock_model, device):
        model_on_device = mock_model.to(device)
        assert model_on_device is not None


class TestModelKwargs:
    def test_build_model_with_args(self, mock_args):
        from src.models.model_factory import _build_model

        mock_model_class = MagicMock()
        _build_model(mock_model_class, args=mock_args)

        mock_model_class.assert_called_once()

    def test_build_model_filters_kwargs(self):
        from src.models.model_factory import _build_model

        class DummyModel:
            def __init__(self, param1):
                self.param1 = param1

        model = _build_model(DummyModel, param1="value1", param2="value2")

        assert model.param1 == "value1"


class TestModelRegistry:
    @patch("models.Mirai.model.Mirai")
    @patch("models.ImgFeatAlign.model.ImgFeatAlign")
    @patch("models.LMVNet.model.LMVNet")
    @patch("models.VMRAMAR.model.VMRAMaR")
    @patch("models.OABreaCR.model.OA_BreaCR")
    @patch("src.models.model_factory.build_mammo_reg_net")
    def test_all_supported_models_can_be_accessed(
        self,
        mock_build_reg,
        mock_oabreacr,
        mock_vmramar,
        mock_lmvnet,
        mock_imgfeatalign,
        mock_mirai,
        mock_args_mirai,
        mock_args_imgfeatalign,
        mock_args_lmvnet,
        mock_args_vmramar,
        mock_args_oabreacr,
    ):
        from src.models.model_factory import get_model

        mock_build_reg.return_value = MagicMock()

        mock_mirai.return_value = MagicMock()
        mock_imgfeatalign.return_value = MagicMock()
        mock_lmvnet.return_value = MagicMock()
        mock_vmramar.return_value = MagicMock()
        mock_oabreacr.return_value = MagicMock()

        assert get_model("Mirai", args=mock_args_mirai) is not None
        assert get_model("ImgFeatAlign", args=mock_args_imgfeatalign, path_saved_reg_model="x.pth") is not None
        assert get_model("LMV-Net", args=mock_args_lmvnet, path_saved_reg_model="x.pth") is not None
        assert get_model("VMRA-MaR", args=mock_args_vmramar) is not None
        assert get_model("OA-BreaCR", args=mock_args_oabreacr) is not None


class TestModelWithSpecificConfigs:
    def test_mirai_training_with_specific_config(self, mock_args_mirai, mock_config_mirai):
        assert mock_args_mirai.model == "Mirai"
        assert mock_config_mirai["num_images"] == 4
        assert mock_args_mirai.num_images == 4
        assert not mock_config_mirai["freeze_image_encoder"]

    def test_imgfeatalign_training_with_specific_config(self, mock_args_imgfeatalign, mock_config_imgfeatalign):
        assert mock_args_imgfeatalign.model == "ImgFeatAlign"
        assert mock_args_imgfeatalign.path_saved_reg_model is not None
        assert mock_config_imgfeatalign["use_deformation"]

    def test_lmvnet_training_with_specific_config(self, mock_args_lmvnet, mock_config_lmvnet):
        assert mock_args_lmvnet.model == "LMV-Net"
        assert mock_args_lmvnet.num_views == 2
        assert mock_args_lmvnet.num_timepoints == 2
        assert mock_config_lmvnet["use_attention"]

    def test_vmramar_training_with_specific_config(self, mock_args_vmramar, mock_config_vmramar):
        assert mock_args_vmramar.model == "VMRA-MaR"
        assert mock_args_vmramar.num_images == 4
        assert mock_config_vmramar["use_asymmetry_detector"]
        assert mock_config_vmramar["use_longitudinal_tracker"]

    def test_oabreacr_training_with_specific_config(self, mock_args_oabreacr, mock_config_oabreacr):
        assert mock_args_oabreacr.model == "OA-BreaCR"
        assert mock_args_oabreacr.num_views == 1
        assert mock_args_oabreacr.num_timepoints == 2
        assert mock_config_oabreacr["use_deformation_field"]