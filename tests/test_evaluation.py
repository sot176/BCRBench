import json

import numpy as np
import pytest

import src.utils.utils as utils_module

from src.utils.utils import (
    bootstrap_auc,
    bootstrap_auc_by_cancer_type,
    bootstrap_auc_by_density,
    bootstrap_auc_by_race,
    bootstrap_c_index_by_cancer_type,
    bootstrap_c_index_by_density,
    bootstrap_c_index_by_race,
    bootstrap_confidence_interval,
    compute_auc_by_density_category,
    compute_auc_x_year_auc,
    compute_c_index_by_density,
    auc_by_cancer_type,
    map_density,
)

def make_auc_dataset():
    """
    Small deterministic dataset:
    - two positives at time 1
    - two negatives censored at time 5
    - perfectly separated predictions
    """
    event_times = np.array([1, 1, 5, 5])
    event_observed = np.array([1, 1, 0, 0])
    predictions = np.array(
        [
            [0.95, 0.95, 0.95, 0.95, 0.95],
            [0.90, 0.90, 0.90, 0.90, 0.90],
            [0.10, 0.10, 0.10, 0.10, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.05],
        ]
    )
    return event_times, predictions, event_observed


def make_grouped_binary_dataset(n_per_group=10):
    """
    Creates two balanced groups with clearly different prediction means.
    Useful for subgroup bootstrapping tests.
    """
    n = n_per_group * 2
    event_times = np.full(n, 2)
    event_observed = np.array(([1, 0] * (n // 2))[:n])
    cancer_categories = np.array([0] * n_per_group + [1] * n_per_group)
    predictions = np.array([0.0] * n_per_group + [1.0] * n_per_group)
    return event_times, predictions, event_observed, cancer_categories


@pytest.mark.evaluation
class TestUtilityFunctions:
    def test_map_density_valid_and_invalid(self):
        assert map_density(1) == "A"
        assert map_density(2) == "B"
        assert map_density(3) == "C"
        assert map_density(4) == "D"
        assert map_density(5) == "NA"
        assert map_density(-1) == "NA"
        assert map_density(None) == "NA"

    def test_bootstrap_confidence_interval_constant_data(self):
        low, high = bootstrap_confidence_interval(
            [3.0, 3.0, 3.0, 3.0],
            num_samples=50,
            confidence_level=0.95,
        )

        assert low == pytest.approx(3.0)
        assert high == pytest.approx(3.0)


@pytest.mark.evaluation
class TestAUCFunctions:
    def test_compute_auc_x_year_auc_perfect_separation(self):
        event_times, predictions, event_observed = make_auc_dataset()

        with pytest.warns(Warning):
            result = compute_auc_x_year_auc(
                predictions,
                event_times,
                event_observed,
            )

        assert set(result.keys()) == {0, 1, 2, 3, 4}
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(1.0)

    def test_bootstrap_auc_returns_expected_structure(self):
        np.random.seed(0)
        event_times, predictions, event_observed = make_auc_dataset()

        summary, raw = bootstrap_auc(
            event_times,
            predictions,
            event_observed,
            n_bootstrap=10,
        )

        assert set(summary.keys()) == {f"Year {i}" for i in range(1, 6)}
        assert set(raw.keys()) == {f"Year {i}" for i in range(1, 6)}

        for year_key, (mean_auc, ci) in summary.items():
            assert isinstance(ci, tuple)
            assert len(ci) == 2
            assert len(raw[year_key]) == 10
            if np.isfinite(mean_auc):
                assert 0.0 <= mean_auc <= 1.0

    def test_compute_auc_handles_single_class_case(self):
        probs = np.zeros((5, 5))
        censor_times = np.ones(5)
        golds = np.zeros(5)

        with pytest.warns(Warning):
            result = compute_auc_x_year_auc(probs, censor_times, golds)

        assert isinstance(result, dict)
        assert all(np.isnan(v) for v in result.values())


@pytest.mark.evaluation
class TestCancerMetrics:
    def test_auc_by_cancer_type_returns_nested_year_dicts(self):
        n = 42
        rng = np.random.default_rng(0)

        result = auc_by_cancer_type(
            rng.integers(1, 6, n),
            rng.random((n, 5)),
            np.array(([1, 0] * 21)),
            np.repeat(np.arange(7), 6),
        )

        assert isinstance(result, dict)
        assert 0 in result
        assert "Year 1" in result[0]

    def test_bootstrap_auc_by_cancer_type_returns_all_categories(self):
        rng = np.random.default_rng(1)
        n = 70

        result = bootstrap_auc_by_cancer_type(
            rng.integers(1, 6, n),
            rng.random((n, 5)),
            np.array(([1, 0] * 35)),
            np.repeat(np.arange(7), 10),
            n_bootstrap=5,
        )

        assert set(result.keys()) == set(range(7))
        assert "Year 1" in result[0]

    def test_bootstrap_c_index_by_cancer_type_respects_category_subsets(self, monkeypatch):
        event_times, predictions, event_observed, cancer_categories = make_grouped_binary_dataset()

        def identity_resample(arr, replace=True, n_samples=None):
            return np.asarray(arr)

        def fake_c_index(event_times, predictions, event_observed, censoring_dist):
            return float(np.mean(predictions))

        monkeypatch.setattr(utils_module, "resample", identity_resample)
        monkeypatch.setattr(utils_module, "concordance_index_ipcw", fake_c_index)

        result = bootstrap_c_index_by_cancer_type(
            event_times,
            predictions,
            event_observed,
            cancer_categories,
            censoring_dist=None,
            n_bootstrap=1,
        )

        # These assertions catch the bug where the function bootstraps from
        # the full dataset instead of the category subset.
        assert result[0][0] == pytest.approx(0.0)
        assert result[1][0] == pytest.approx(1.0)


@pytest.mark.evaluation
class TestDensityMetrics:
    def test_compute_auc_by_density_category_returns_all_density_keys(self):
        n = 40
        rng = np.random.default_rng(2)

        result = compute_auc_by_density_category(
            rng.random((n, 5)),
            rng.integers(1, 6, n),
            np.array(([1, 0] * 20)),
            np.tile([1, 2, 3, 4], 10),
        )

        assert set(result.keys()) == {"A", "B", "C", "D"}
        assert set(result["A"].keys()) == {0, 1, 2, 3, 4}

    def test_compute_c_index_by_density_returns_all_density_keys(self, monkeypatch):
        n = 40
        rng = np.random.default_rng(3)

        monkeypatch.setattr(
            utils_module,
            "concordance_index_ipcw",
            lambda event_times, predictions, event_observed, censoring_dist: 0.75,
        )

        result = compute_c_index_by_density(
            rng.integers(1, 6, n),
            rng.random(n),
            np.array(([1, 0] * 20)),
            np.tile([1, 2, 3, 4], 10),
            censoring_dist=None,
        )

        assert result == {"A": 0.75, "B": 0.75, "C": 0.75, "D": 0.75}

    def test_bootstrap_auc_by_density_returns_nested_summary(self):
        n = 40
        rng = np.random.default_rng(4)

        result = bootstrap_auc_by_density(
            rng.integers(1, 6, n),
            rng.random((n, 5)),
            np.array(([1, 0] * 20)),
            np.tile([1, 2, 3, 4], 10),
            n_bootstrap=5,
        )

        assert set(result.keys()) == {"A", "B", "C", "D"}
        assert "Year 1" in result["A"]

    def test_bootstrap_c_index_by_density_saves_json_and_matches_summary(self, tmp_path, monkeypatch):
        n = 40
        rng = np.random.default_rng(5)

        monkeypatch.setattr(
            utils_module,
            "concordance_index_ipcw",
            lambda event_times, predictions, event_observed, censoring_dist: 0.6,
        )

        summary, raw = bootstrap_c_index_by_density(
            rng.integers(1, 6, n),
            rng.random(n),
            np.array(([1, 0] * 20)),
            np.tile([1, 2, 3, 4], 10),
            censoring_dist=None,
            n_bootstrap=5,
            save_json_path=tmp_path,
        )

        saved_file = tmp_path / "mbox_plots_c_index_density_results.json"
        assert saved_file.exists()

        with saved_file.open("r") as f:
            payload = json.load(f)

        assert set(payload.keys()) == {"A", "B", "C", "D"}
        assert isinstance(summary, dict)
        assert isinstance(raw, dict)
        assert summary["A"][0] == pytest.approx(0.6)
        assert len(raw["A"]) == 5


@pytest.mark.evaluation
class TestRaceMetrics:
    def test_bootstrap_auc_by_race_normalizes_blank_to_unknown(self):
        n = 16
        event_times = np.array([1, 1, 5, 5] * 4)
        predictions = np.tile(
            np.array(
                [
                    [0.95, 0.95, 0.95, 0.95, 0.95],
                    [0.90, 0.90, 0.90, 0.90, 0.90],
                    [0.10, 0.10, 0.10, 0.10, 0.10],
                    [0.05, 0.05, 0.05, 0.05, 0.05],
                ]
            ),
            (4, 1),
        )
        event_observed = np.array([1, 1, 0, 0] * 4)
        races = np.array(["", None, "Unknown", "Unknown"] * 4, dtype=object)

        result = bootstrap_auc_by_race(
            event_times,
            predictions,
            event_observed,
            races,
            n_bootstrap=5,
        )

        assert "Unknown" in result
        assert "Year 1" in result["Unknown"]

    def test_bootstrap_c_index_by_race_saves_json(self, tmp_path, monkeypatch):
        n = 16
        event_times = np.arange(1, n + 1)
        predictions = np.linspace(0.1, 0.9, n)
        event_observed = np.array(([1, 0] * 8))
        races = np.array(["Caucasian or White"] * 8 + ["Asian"] * 8, dtype=object)

        monkeypatch.setattr(
            utils_module,
            "concordance_index_ipcw",
            lambda event_times, predictions, event_observed, censoring_dist: 0.7,
        )

        summary, raw = bootstrap_c_index_by_race(
            event_times,
            predictions,
            event_observed,
            races,
            censoring_dist=None,
            n_bootstrap=5,
            save_json_path=tmp_path,
        )

        saved_file = tmp_path / "c_index_race_bootstrap_samples.json"
        assert saved_file.exists()

        with saved_file.open("r") as f:
            payload = json.load(f)

        assert "Caucasian or White" in payload
        assert "Asian" in payload
        assert summary["Caucasian or White"][0] == pytest.approx(0.7)
        assert len(raw["Asian"]) == 5
