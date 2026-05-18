import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import types

from src.train.train_utils import (
    save_checkpoint,
    load_checkpoint,
)
from src.evaluate.test_utils import load_model

# ------------------------------------------------------------
# 1. TEST checkpoint()
# ------------------------------------------------------------

def test_checkpoint_saves_state_dict(temp_dir):
    model = nn.Linear(10, 5)
    path = temp_dir / "model.pth"

    save_checkpoint(model, path)

    assert path.exists()

    loaded = torch.load(path)
    assert isinstance(loaded, dict)

    assert "weight" in loaded
    assert "bias" in loaded


# ------------------------------------------------------------
# 2. TEST load_model()
# ------------------------------------------------------------

@patch("src.checkpoint_utils.cfg", {
    "paths": {
        "csaw_path_saved_reg_model": "/csaw",
        "embed_path_saved_reg_model": "/embed"
    }
})
@patch("src.checkpoint_utils.get_model")
@patch("src.checkpoint_utils.torch.load")
def test_load_model_embed(mock_torch_load, mock_get_model, temp_dir):

    args = types.SimpleNamespace(
        dataset="EMBED",
        model="TestModel",
        finetune_all=True
    )

    fake_model = MagicMock()
    fake_model.eval.return_value = "EVAL_MODEL"
    fake_model.load_state_dict = MagicMock()

    mock_get_model.return_value = fake_model

    mock_torch_load.return_value = {
        "model": {
            "module.layer.weight": torch.tensor([1.0])
        }
    }

    result = load_model(args, "dummy.pth")

    mock_get_model.assert_called_once()

    loaded_state = fake_model.load_state_dict.call_args[0][0]

    assert "module.layer.weight" not in loaded_state
    assert "layer.weight" in loaded_state

    assert result == "EVAL_MODEL"


# ------------------------------------------------------------
# 3. TEST save_checkpoint()
# ------------------------------------------------------------

def test_save_checkpoint(temp_dir):
    from src.main_train import save_checkpoint

    accelerator = MagicMock()
    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    warmup = MagicMock()

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup,
        epoch=2,
        global_step=50,
        best_c_index=0.77,
        path=str(temp_dir / "ckpt.pth"),
    )

    accelerator.save.assert_called_once()

    saved = accelerator.save.call_args[0][0]

    assert saved["epoch"] == 2
    assert saved["global_step"] == 50
    assert saved["best_c_index"] == 0.77

    assert "model" in saved
    assert "optimizer" in saved
    assert "scheduler" in saved
    assert "warmup_scheduler" in saved


# ------------------------------------------------------------
# 4. TEST load_checkpoint()
# ------------------------------------------------------------

def test_load_checkpoint(temp_dir):
    ckpt_path = temp_dir / "ckpt.pth"

    fake_ckpt = {
        "epoch": 4,
        "global_step": 200,
        "best_c_index": 0.91,
        "model": {"w": torch.tensor([1.0])},
        "optimizer": {},
        "scheduler": {},
        "warmup_scheduler": {}
    }

    torch.save(fake_ckpt, ckpt_path)

    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    warmup_scheduler = MagicMock()

    accelerator = MagicMock()
    accelerator.unwrap_model.return_value = model

    result = load_checkpoint(
        str(ckpt_path),
        model,
        optimizer,
        scheduler,
        warmup_scheduler,
        accelerator,
    )

    model.load_state_dict.assert_called_once()
    optimizer.load_state_dict.assert_called_once()
    scheduler.load_state_dict.assert_called_once()
    warmup_scheduler.load_state_dict.assert_called_once()

    assert result == (5, 200, 0.91)