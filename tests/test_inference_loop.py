import pytest
import torch
from types import SimpleNamespace
from pathlib import Path


# Adjust this import to your project structure
from src.evaluate.test_risk_prediction import test_risk
 

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        dataset="CSAW",
        model="dummy_model",
        id_training="1",
        path_test_folder=str(tmp_path),
    )


@pytest.fixture
def mock_model():
    import torch.nn as nn

    class MockModel(nn.Module):
        def forward(self, batch):
            return torch.randn(batch["images"].shape[0], 1)

        def get_primary_risk_head(self, outputs):
            return outputs

    return MockModel()

 

@pytest.fixture
def accelerator():
    class DummyAccelerator:
        def __init__(self):
            self.is_main_process = True

        def prepare(self, model, loader):
            return model, loader

        def unwrap_model(self, model):
            return model

        def gather(self, x):
            return x

    return DummyAccelerator()


# -------------------------
# Mock external dependencies
# -------------------------

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, mock_model):

    import src.evaluate.test_risk_prediction as m

    monkeypatch.setattr(
        m,
        "load_model",
        lambda args, path: mock_model
    )

    monkeypatch.setattr(m, "create_logger", lambda *a, **k: None)
    monkeypatch.setattr(m, "save_model_results_to_file", lambda *a, **k: None)

    monkeypatch.setattr(m, "bootstrap_c_index", lambda *a, **k: (0.75, None, []))
    monkeypatch.setattr(m, "bootstrap_auc", lambda *a, **k: ({"1": (0.8, 0.0)}, {}))

    monkeypatch.setattr(m, "bootstrap_auc_by_density", lambda *a, **k: {})
    monkeypatch.setattr(m, "bootstrap_c_index_by_density", lambda *a, **k: ({}, []))

    monkeypatch.setattr(m, "bootstrap_auc_by_cancer_type", lambda *a, **k: {})
    monkeypatch.setattr(m, "bootstrap_c_index_by_cancer_type", lambda *a, **k: ({}, []))

    monkeypatch.setattr(m, "bootstrap_auc_by_race", lambda *a, **k: {})
    monkeypatch.setattr(m, "bootstrap_c_index_by_race", lambda *a, **k: ({}, []))

    monkeypatch.setattr(m, "get_censoring_dist", lambda t, e: None)


# -------------------------
# MAIN TEST
# -------------------------

def test_test_risk_runs_end_to_end(
    args,
    test_loader,
    accelerator,
    tmp_path,
):
    """
    Smoke test: ensures inference pipeline runs without crashing.
    """

    test_risk(
        args,
        test_loader=test_loader,
        path_model="dummy.ckpt",
        accelerator=accelerator,
    )
 
    assert tmp_path.exists()