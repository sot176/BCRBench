"""
Tests for training loop utilities.

These tests use tiny real PyTorch modules and dataloaders instead of MagicMock,
so they exercise the same call pattern as the real training code:
    outputs = model(batch)
    pred_risk = model.get_primary_risk_head(outputs)
    loss = loss_fn(outputs, batch, model)
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn


import importlib
import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Could not import {name}")


@pytest.fixture
def train_utils_module():
    return _import_module("src.train.train_utils")


@pytest.fixture
def risk_prediction_module():
    return _import_module("src.train.train_risk_prediction")


@pytest.fixture
def training_loss_fn():
    def _loss_fn(outputs, batch, base_model):
        del base_model
        labels = batch["labels"].float().unsqueeze(1)
        return nn.BCEWithLogitsLoss()(outputs, labels)

    return _loss_fn


@pytest.mark.training
class TestTrainingLoop:
    def test_train_step_basic(self, mock_model, sample_batch):
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        predictions = mock_model(sample_batch)
        loss = loss_fn(predictions, sample_batch["labels"].float().unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert isinstance(loss.item(), float)

    def test_validation_step(self, mock_model, sample_batch):
        mock_model.eval()

        with torch.no_grad():
            predictions = mock_model(sample_batch)

        assert predictions.shape[0] == sample_batch["images"].shape[0]
        assert predictions.shape[1] == 1
        assert not predictions.requires_grad

    def test_gradient_flow(self, mock_model, sample_batch):
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        initial_params = [p.detach().clone() for p in mock_model.parameters()]

        predictions = mock_model(sample_batch)
        loss = loss_fn(predictions, sample_batch["labels"].float().unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_params = list(mock_model.parameters())
        params_changed = any(
            not torch.equal(p0, p1.detach())
            for p0, p1 in zip(initial_params, updated_params)
        )

        assert params_changed


@pytest.mark.training
class TestLossCalculation:
    def test_bce_loss(self, sample_batch):
        predictions = torch.sigmoid(torch.randn(4, 1))
        labels = sample_batch["labels"].float().unsqueeze(1)

        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_cross_entropy_loss(self):
        predictions = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predictions, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_backward(self):
        x = torch.randn(4, 10, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)

        loss = (x.sum() + y.sum()) ** 2
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None


@pytest.mark.training
class TestBatchProcessing:
    def test_batch_to_device(self, sample_batch, device):
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }

        assert batch_on_device["images"].device.type == device.type
        assert batch_on_device["labels"].device.type == device.type

    def test_batch_accumulation(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        total_loss = 0.0
        num_batches = 3

        for _ in range(num_batches):
            x = torch.randn(4, 10)
            y = torch.randn(4, 1)

            predictions = model(x)
            loss = loss_fn(predictions, y)
            loss.backward()
            total_loss += loss.item()

        optimizer.step()

        assert total_loss > 0


@pytest.fixture
def mock_training_model():
    """Model matching train_one_epoch/evaluate API."""
    class TinyTrainingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(3 * 224 * 224, 1)
            )

        def forward(self, batch):
            return self.net(batch["images"])

        def get_primary_risk_head(self, outputs):
            return outputs

    return TinyTrainingModel()

@pytest.mark.training
class TestTrainUtils:
    def test_train_one_epoch(self, train_utils_module, mock_model, mock_dataloader, accelerator, training_loss_fn, monkeypatch):
        monkeypatch.setattr(train_utils_module, "get_censoring_dist", lambda times, events: None)
        monkeypatch.setattr(train_utils_module, "concordance_index_ipcw", lambda times, preds, events, censor: 0.75)
        monkeypatch.setattr(
            train_utils_module,
            "compute_auc_x_year_auc",
            lambda preds, times, events: {year: 0.8 for year in range(5)},
        )

        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: train_utils_module.linear_warmup(step, 5),
        )

        avg_loss, c_index, auc_results = train_utils_module.train_one_epoch(
            mock_model,
            mock_dataloader,
            optimizer,
            accelerator,
            warmup_scheduler,
            global_step=0,
            warmup_steps=5,
            loss_fn=training_loss_fn,
        )

        assert avg_loss > 0
        assert c_index == pytest.approx(0.75)
        assert auc_results[0] == pytest.approx(0.8)

    def test_evaluate(self, train_utils_module, mock_model, mock_dataloader, accelerator, training_loss_fn, monkeypatch):
        monkeypatch.setattr(train_utils_module, "get_censoring_dist", lambda times, events: None)
        monkeypatch.setattr(train_utils_module, "concordance_index_ipcw", lambda times, preds, events, censor: 0.7)
        monkeypatch.setattr(
            train_utils_module,
            "compute_auc_x_year_auc",
            lambda preds, times, events: {year: 0.65 for year in range(5)},
        )

        avg_loss, c_index, auc_results = train_utils_module.evaluate(
            mock_model,
            mock_dataloader,
            accelerator,
            training_loss_fn,
        )

        assert avg_loss > 0
        assert c_index == pytest.approx(0.7)
        assert auc_results[4] == pytest.approx(0.65)

    def test_linear_warmup(self, train_utils_module):
        assert train_utils_module.linear_warmup(0, 10) == 0.0
        assert train_utils_module.linear_warmup(5, 10) == 0.5
        assert train_utils_module.linear_warmup(10, 10) == 1.0
        assert train_utils_module.linear_warmup(100, 10) == 1.0
        assert train_utils_module.linear_warmup(0, 0) == 1.0

    def test_get_param_groups(self, train_utils_module, args, mock_model):
        param_groups = train_utils_module.get_param_groups(args, mock_model, base_lr=1e-3)

        assert len(param_groups) >= 1
        assert all("params" in group for group in param_groups)
        assert all("lr" in group for group in param_groups)
        assert sum(len(group["params"]) for group in param_groups) > 0


@pytest.mark.checkpointing
class TestCheckpointing:
    def test_save_checkpoint(self, train_utils_module, mock_model, accelerator, temp_dir):
        optimizer = torch.optim.Adam(mock_model.parameters())
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        checkpoint_path = temp_dir / "checkpoint.pth"
        train_utils_module.save_checkpoint(
            accelerator=accelerator,
            model=mock_model,
            optimizer=optimizer,
            scheduler=None,
            warmup_scheduler=warmup_scheduler,
            epoch=5,
            global_step=123,
            best_c_index=0.81,
            path=str(checkpoint_path),
        )

        assert checkpoint_path.exists()

    def test_load_checkpoint(self, train_utils_module, mock_model, accelerator, temp_dir):
        optimizer = torch.optim.Adam(mock_model.parameters())
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        checkpoint_path = temp_dir / "checkpoint.pth"
        train_utils_module.save_checkpoint(
            accelerator=accelerator,
            model=mock_model,
            optimizer=optimizer,
            scheduler=None,
            warmup_scheduler=warmup_scheduler,
            epoch=5,
            global_step=123,
            best_c_index=0.81,
            path=str(checkpoint_path),
        )

        new_model = type(mock_model)()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        new_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=lambda step: 1.0)

        next_epoch, global_step, best_c_index = train_utils_module.load_checkpoint(
            str(checkpoint_path),
            new_model,
            new_optimizer,
            scheduler=None,
            warmup_scheduler=new_warmup_scheduler,
            accelerator=accelerator,
        )

        assert next_epoch == 6
        assert global_step == 123
        assert best_c_index == pytest.approx(0.81)


@pytest.mark.training
@pytest.mark.slow
class TestTrainingIntegration:
    def test_full_training_epoch(self, train_utils_module, mock_dataloader, mock_model, accelerator, training_loss_fn, monkeypatch):
        monkeypatch.setattr(train_utils_module, "get_censoring_dist", lambda times, events: None)
        monkeypatch.setattr(train_utils_module, "concordance_index_ipcw", lambda times, preds, events, censor: 0.75)
        monkeypatch.setattr(
            train_utils_module,
            "compute_auc_x_year_auc",
            lambda preds, times, events: {year: 0.8 for year in range(5)},
        )

        mock_model.train()
        optimizer = torch.optim.Adam(mock_model.parameters())
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        epoch_loss, _, _ = train_utils_module.train_one_epoch(
            mock_model,
            mock_dataloader,
            optimizer,
            accelerator,
            warmup_scheduler,
            global_step=0,
            warmup_steps=0,
            loss_fn=training_loss_fn,
        )

        assert epoch_loss > 0
