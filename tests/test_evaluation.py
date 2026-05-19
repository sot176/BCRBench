"""
Tests for evaluation metrics and utilities.

Tests the utils/c_index.py and evaluation functions:
- concordance_index_ipcw()
- bootstrap_c_index()
- get_censoring_dist()
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.mark.evaluation
class TestCensoringDistribution:
    """Test censoring distribution estimation."""
    
    def test_get_censoring_dist_basic(self):
        """Test basic censoring distribution calculation."""
        from utils.c_index import get_censoring_dist
        
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_observed = np.array([1, 1, 0, 1, 0])
        
        censoring_dist = get_censoring_dist(times, event_observed)
        
        assert isinstance(censoring_dist, dict)
        assert len(censoring_dist) > 0
        assert all(0 < v <= 1 for v in censoring_dist.values())
    
    def test_censoring_dist_decreasing(self):
        """Test that censoring probabilities decrease over time."""
        from utils.c_index import get_censoring_dist
        
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_observed = np.array([0, 0, 0, 0, 0])  # All censored
        
        censoring_dist = get_censoring_dist(times, event_observed)
        
        # Probabilities should decrease with time
        sorted_times = sorted(censoring_dist.keys())
        probs = [censoring_dist[t] for t in sorted_times]
        
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]
    
    def test_censoring_dist_range(self):
        """Test that censoring distribution values are in valid range."""
        from utils.c_index import get_censoring_dist
        
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_observed = np.array([1, 0, 1, 0, 1])
        
        censoring_dist = get_censoring_dist(times, event_observed)
        
        for value in censoring_dist.values():
            assert 0 < value <= 1


@pytest.mark.evaluation
class TestConcordanceIndex:
    """Test concordance index calculation."""
    
    def test_concordance_index_basic(self, sample_event_times, sample_predictions, 
                                     sample_event_observed, censoring_dist):
        """Test basic C-index calculation."""
        from utils.c_index import concordance_index_ipcw
        
        c_index = concordance_index_ipcw(
            sample_event_times,
            sample_predictions,
            sample_event_observed,
            censoring_dist
        )
        
        assert isinstance(c_index, float)
        assert 0 <= c_index <= 1
    
    def test_concordance_index_perfect_predictions(self, censoring_dist):
        """Test C-index with perfect predictions."""
        from utils.c_index import concordance_index_ipcw
        
        # Create perfectly predictive data
        event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Matches order
        event_observed = np.array([1, 1, 1, 1, 1])
        
        c_index = concordance_index_ipcw(
            event_times,
            predictions,
            event_observed,
            censoring_dist
        )
        
        # Should be high for perfect predictions
        assert c_index > 0.5
    
    def test_concordance_index_bad_predictions(self, censoring_dist):
        """Test C-index with random predictions."""
        from utils.c_index import concordance_index_ipcw
        
        np.random.seed(42)
        event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.random.rand(5)  # Random
        event_observed = np.array([1, 1, 1, 1, 1])
        
        c_index = concordance_index_ipcw(
            event_times,
            predictions,
            event_observed,
            censoring_dist
        )
        
        assert isinstance(c_index, float)
        assert 0 <= c_index <= 1
    
    def test_concordance_index_with_censoring(self):
        """Test C-index with censored data."""
        from utils.c_index import concordance_index_ipcw, get_censoring_dist
        
        event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_observed = np.array([1, 0, 1, 0, 1])  # Some censored
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        censoring_dist = get_censoring_dist(event_times, event_observed)
        
        c_index = concordance_index_ipcw(
            event_times,
            predictions,
            event_observed,
            censoring_dist
        )
        
        assert isinstance(c_index, float)
        assert 0 <= c_index <= 1
    

    def test_concordance_index_no_pairs(self, censoring_dist):
        """Test error handling when no admissible pairs."""
        from utils.c_index import concordance_index_ipcw
        
        event_times = np.array([1.0, 1.0, 1.0])
        predictions = np.array([0.1, 0.1, 0.1])
        event_observed = np.array([0, 0, 0])  # All censored at same time
        
        with pytest.raises(ZeroDivisionError):
            concordance_index_ipcw(
                event_times,
                predictions,
                event_observed,
                censoring_dist
            )


@pytest.mark.evaluation
class TestBootstrapCIndex:
    """Test bootstrap confidence intervals for C-index."""
    
    def test_bootstrap_c_index_basic(self, sample_event_times, sample_predictions,
                                      sample_event_observed):
        """Test basic bootstrap C-index."""
        from utils.utils import bootstrap_c_index
        from utils.c_index import get_censoring_dist
        
        censoring_dist = get_censoring_dist(sample_event_times, sample_event_observed)
        
        mean_c_index, ci = bootstrap_c_index(
            sample_event_times,
            sample_predictions,
            sample_event_observed,
            censoring_dist,
            n_bootstrap=100,
            alpha=0.05
        )
        
        assert isinstance(mean_c_index, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert 0 <= mean_c_index <= 1
        assert ci[0] <= mean_c_index <= ci[1]
    
    def test_bootstrap_ci_bounds(self, sample_event_times, sample_predictions,
                                 sample_event_observed):
        """Test bootstrap confidence interval bounds."""
        from utils.utils import bootstrap_c_index
        from utils.c_index import get_censoring_dist
        
        censoring_dist = get_censoring_dist(sample_event_times, sample_event_observed)
        
        mean_c_index, (lower, upper) = bootstrap_c_index(
            sample_event_times,
            sample_predictions,
            sample_event_observed,
            censoring_dist,
            n_bootstrap=100
        )
        
        assert lower < upper
        assert lower >= 0
        assert upper <= 1
    
    def test_bootstrap_sample_count(self, sample_event_times, sample_predictions,
                                    sample_event_observed):
        """Test bootstrap with different sample counts."""
        from utils.utils import bootstrap_c_index
        from utils.c_index import get_censoring_dist
        
        censoring_dist = get_censoring_dist(sample_event_times, sample_event_observed)
        
        # Test with different bootstrap sizes
        for n_bootstrap in [10, 100, 500]:
            mean_c_index, ci = bootstrap_c_index(
                sample_event_times,
                sample_predictions,
                sample_event_observed,
                censoring_dist,
                n_bootstrap=n_bootstrap
            )
            assert isinstance(mean_c_index, float)
            assert 0 <= mean_c_index <= 1
    
    def test_bootstrap_alpha_levels(self, sample_event_times, sample_predictions,
                                     sample_event_observed):
        """Test bootstrap with different alpha levels."""
        from utils.utils import bootstrap_c_index
        from utils.c_index import get_censoring_dist
        
        censoring_dist = get_censoring_dist(sample_event_times, sample_event_observed)
        
        # Test with different significance levels
        for alpha in [0.01, 0.05, 0.10]:
            mean_c_index, (lower, upper) = bootstrap_c_index(
                sample_event_times,
                sample_predictions,
                sample_event_observed,
                censoring_dist,
                n_bootstrap=100,
                alpha=alpha
            )
            assert lower < upper


@pytest.mark.evaluation
class TestMetricsDataTypes:
    """Test handling of different data types."""
    
    def test_concordance_index_with_lists(self, censoring_dist):
        """Test C-index with list inputs."""
        from utils.c_index import concordance_index_ipcw
        
        event_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        predictions = [0.1, 0.2, 0.3, 0.4, 0.5]
        event_observed = [1, 1, 1, 1, 1]
        
        c_index = concordance_index_ipcw(
            event_times,
            predictions,
            event_observed,
            censoring_dist
        )
        
        assert isinstance(c_index, float)
        assert 0 <= c_index <= 1
    
    def test_concordance_index_with_arrays(self, censoring_dist):
        """Test C-index with numpy array inputs."""
        from utils.c_index import concordance_index_ipcw
        
        event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        event_observed = np.array([1, 1, 1, 1, 1])
        
        c_index = concordance_index_ipcw(
            event_times,
            predictions,
            event_observed,
            censoring_dist
        )
        
        assert isinstance(c_index, float)
        assert 0 <= c_index <= 1


@pytest.mark.evaluation
@pytest.mark.slow
class TestMetricsStability:
    """Test stability of metrics across multiple runs."""
    
    def test_bootstrap_reproducibility(self, sample_event_times, sample_predictions,
                                       sample_event_observed):
        """Test bootstrap reproducibility with fixed seed."""
        from utils.utils import bootstrap_c_index
        from utils.c_index import get_censoring_dist
        
        np.random.seed(42)
        censoring_dist = get_censoring_dist(sample_event_times, sample_event_observed)
        
        np.random.seed(42)
        result1 = bootstrap_c_index(
            sample_event_times,
            sample_predictions,
            sample_event_observed,
            censoring_dist,
            n_bootstrap=50
        )
        
        np.random.seed(42)
        result2 = bootstrap_c_index(
            sample_event_times,
            sample_predictions,
            sample_event_observed,
            censoring_dist,
            n_bootstrap=50
        )
        
        assert np.isclose(result1[0], result2[0])
