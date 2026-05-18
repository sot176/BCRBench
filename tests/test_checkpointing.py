import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import types

from src.train.train_utils import (
    save_checkpoint,
    load_checkpoint,
)


# ------------------------------------------------------------
# 1. TEST save_checkpoint()
# ------------------------------------------------------------

def test_save_checkpoint(temp_dir):

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
# 2. TEST load_checkpoint()
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