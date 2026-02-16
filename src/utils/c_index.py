from lifelines import KaplanMeierFitter
from lifelines.utils.btree import _BTree
import numpy as np


def get_censoring_dist(times, event_observed):
    """
    Estimate the censoring distribution using Kaplan-Meier.

    Args:
        times (array-like): Observed survival times.
        event_observed (array-like): Event indicators (1 if event occurred, 0 if censored).

    Returns:
        dict: Mapping from time to probability of being uncensored at that time (P(T > t)).
    """
    times = list(times)
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)
    unique_times = set(times)

    censoring_dist = {
        time: max(kmf.predict(time), 1e-6) for time in unique_times
    }  # Avoid zero censoring probability

    return censoring_dist


def concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist):
    """
    Compute Uno’s C-index using inverse probability of censoring weighting (IPCW).

    Args:
        event_times (array-like): Observed survival times.
        predictions (ndarray): Time-dependent predicted scores (n_samples x n_timepoints).
        event_observed (array-like): Event indicators (1 if event occurred, 0 if censored).
        censoring_dist (dict): P(T > t) from get_censoring_dist().

    Returns:
        float: IPCW-weighted concordance index.
    """
    predicted_scores = 1 - np.asarray(predictions, dtype=float)
    event_times = np.asarray(event_times, dtype=float)


    if event_observed is None:
        event_observed = np.ones_like(event_times)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError("Shape mismatch between event_times and event_observed.")

    num_correct, num_tied, num_pairs = _concordance_summary_statistics(
        event_times, predicted_scores, event_observed, censoring_dist
    )

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct, num_tied, num_pairs):
    if num_pairs == 0:
        raise ZeroDivisionError("No admissible pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _concordance_summary_statistics(event_times, predicted_event_times, event_observed, censoring_dist):
    """
    Helper to compute concordance statistics in O(n log n) time.

    Args:
        event_times (np.ndarray): True event or censor times.
        predicted_event_times (np.ndarray): Model predictions.
        event_observed (np.ndarray): 1 if observed, 0 if censored.
        censoring_dist (dict): Survival probabilities from get_censoring_dist.

    Returns:
        Tuple[int, int, int]: (num_correct, num_tied, num_pairs)
    """
    if not np.any(event_observed):
        return 0, 0, 0

    observed_times = set(event_times)
    died_mask = event_observed.astype(bool)

    # Sort died patients by time
    died_truth = event_times[died_mask]
    died_pred = predicted_event_times[died_mask]
    idx = np.argsort(died_truth)
    died_truth = died_truth[idx]
    died_pred = died_pred[idx]

    # Sort censored patients by time
    censored_truth = event_times[~died_mask]
    censored_pred = predicted_event_times[~died_mask]
    idx = np.argsort(censored_truth)
    censored_truth = censored_truth[idx]
    censored_pred = censored_pred[idx]

    # Prepare data structures
    times_to_compare = {
        time: _BTree(np.unique(died_pred[:, int(time)])) for time in observed_times
    }

    died_ix = 0
    censored_ix = 0
    num_pairs = num_correct = num_tied = np.int64(0)

    while True:
        has_more_died = died_ix < len(died_truth)
        has_more_censored = censored_ix < len(censored_truth)

        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(
                censored_truth, censored_pred, censored_ix, times_to_compare, censoring_dist
            )
            censored_ix = next_ix

        elif has_more_died:
            pairs, correct, tied, next_ix, weight = _handle_pairs(
                died_truth, died_pred, died_ix, times_to_compare, censoring_dist
            )

            for pred in died_pred[died_ix:next_ix]:
                for time in observed_times:
                    times_to_compare[time].insert(pred[int(time)])

            died_ix = next_ix

        else:
            break

        num_pairs += pairs * weight
        num_correct += correct * weight
        num_tied += tied * weight

    return num_correct, num_tied, num_pairs


def _handle_pairs(truth, pred, start_ix, times_to_compare, censoring_dist):
    """
    Handles pairs with the same time to efficiently update stats.

    Args:
        truth (np.ndarray): True times.
        pred (np.ndarray): Predicted scores.
        start_ix (int): Starting index.
        times_to_compare (dict): Time -> _BTree of previous predictions.
        censoring_dist (dict): Censoring survival probabilities.

    Returns:
        Tuple[int, int, int, int, float]: (pairs, correct, tied, next_ix, weight)
    """
    time = truth[start_ix]
    weight = 1.0 / (censoring_dist[time] ** 2)
    next_ix = start_ix

    while next_ix < len(truth) and truth[next_ix] == time:
        next_ix += 1

    count_at_time = next_ix - start_ix
    btree = times_to_compare[time]
    pairs = len(btree) * count_at_time
    correct = tied = np.int64(0)

    for i in range(start_ix, next_ix):
        # shift because predictions are 0-based, times are 1-based
        idx = int(time) - 1
        rank, tie_count = btree.rank(pred[i][idx])
        correct += rank
        tied += tie_count

    return pairs, correct, tied, next_ix, weight

