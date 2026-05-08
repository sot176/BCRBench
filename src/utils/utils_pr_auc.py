import os
import warnings
import json
from sklearn import metrics
from sklearn.utils import resample
import numpy as np

from .c_index import concordance_index_ipcw

RACES = [
    "Caucasian or White",
    "African American  or Black",
    "Asian",
    "American Indian or Alaskan Native",
    "Native Hawaiian or Other Pacific Islander",
    "Multiple",
    "Unknown",
    "Unavailable or Unreported",
]

RACE_TO_ID = {r: i for i, r in enumerate(RACES)}
ID_TO_RACE = {i: r for r, i in RACE_TO_ID.items()}

def map_density( value):
    """Map numeric density values (1–4) to categorical labels A–D."""
    mapping = {
        1: "A",
        2: "B",
        3: "C",
        4: "D"
    }
    return mapping.get(value, "NA")


def bootstrap_c_index(
    event_times,
    predictions,
    event_observed,
    censoring_dist,
    n_bootstrap=2000,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for concordance index.

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        censoring_dist: Censoring distribution for IPCW calculation
        n_bootstrap: Number of bootstrap samples (default=1000)
        alpha: Significance level for confidence intervals (default=0.05)

    Returns:
        mean_c_index: Mean concordance index over bootstrap samples
        ci: Tuple of lower and upper confidence interval bounds
    """
    c_index_scores = []
    cases = np.where(event_observed == 1)[0]
    controls = np.where(event_observed == 0)[0]

    for _ in range(n_bootstrap):
        sample_cases = resample(cases, replace=True, n_samples=len(cases))
        sample_controls = resample(controls, replace=True, n_samples=len(controls))

        indices = np.concatenate([sample_cases, sample_controls])
        event_times_sample = event_times[indices]
        predictions_sample = predictions[indices]
        event_observed_sample = event_observed[indices]

        c_index = concordance_index_ipcw(
            event_times_sample,
            predictions_sample,
            event_observed_sample,
            censoring_dist,
        )
        c_index_scores.append(c_index)

    lower = np.percentile(c_index_scores, 100 * alpha / 2)
    upper = np.percentile(c_index_scores, 100 * (1 - alpha / 2))

    return np.mean(c_index_scores), (lower, upper),  np.array(c_index_scores)



def bootstrap_confidence_interval(data, num_samples=2000, confidence_level=0.95):
    """
    Calculate the confidence interval using bootstrapping.

    Args:
        data: List or numpy array of metric values.
        num_samples: Number of bootstrap samples to draw (default: 1000).
        confidence_level: Confidence level for the interval (default: 0.95).

    Returns:
        Tuple (lower_bound, upper_bound) representing the confidence interval.
    """
    data = np.array(data)
    bootstrapped_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1 - confidence_level
    lower_bound = np.percentile(bootstrapped_means, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound

def bootstrap_auc(event_times, predictions, event_observed, n_bootstrap=2000, alpha=0.05, max_attempts=50):
    """
    Compute bootstrap confidence intervals for AUC at 5 yearly follow-ups.
    Ensures each year has at least one positive and one negative sample.

    Args:
        event_times: Array of event/censoring times (N,).
        predictions: Array of predicted risk probabilities (N, 5).
        event_observed: Binary array indicating event occurrence (N,).
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level for confidence intervals.
        max_attempts: Max resampling attempts if a sample fails.

    Returns:
        auc_summary: Dict mapping "Year 1" to "Year 5" to tuples (mean_auc, (lower_CI, upper_CI)).
    """
    N = len(event_times)
    auc_results = {f"Year {i+1}": [] for i in range(5)}

    cases = np.where(event_observed == 1)[0]
    controls = np.where(event_observed == 0)[0]

    for _ in range(n_bootstrap):
        sample_cases = resample(cases, replace=True, n_samples=len(cases))
        sample_controls = resample(controls, replace=True, n_samples=len(controls))

        indices = np.concatenate([sample_cases, sample_controls])
        sample_event_times = event_times[indices]
        sample_predictions = predictions[indices]
        sample_event_observed = event_observed[indices]

        yearly_aucs = compute_auc_x_year_auc(sample_predictions, sample_event_times, sample_event_observed)
        
        for year, auc in yearly_aucs.items():
            auc_results[f"Year {year+1}"].append(auc)

    auc_summary = {}
    for year, values in auc_results.items():
        vals = np.array(values)
        mean_auc = np.mean(vals)
        lower = np.percentile(vals, 100 * alpha / 2)
        upper = np.percentile(vals, 100 * (1 - alpha / 2))
        auc_summary[year] = (mean_auc, (lower, upper))

    return auc_summary, auc_results


def compute_auc_x_year_auc(probs, censor_times, golds):
    """
    Compute AUC for each year from 1 to 5 given predicted probabilities, censoring times, and event labels.
    """
    def include_exam_and_determine_label(prob_arr, censor_time, gold, followup):
        valid_pos = gold == 1 and censor_time <= followup
        valid_neg = censor_time >= followup
        included = valid_pos or valid_neg
        label = valid_pos
        return included, label

    aucs_per_year = {}

    for followup in range(5):
        probs_for_eval, golds_for_eval = [], []
        for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
            include, label = include_exam_and_determine_label(prob_arr, censor_time, gold, followup)
            if include:
                probs_for_eval.append(prob_arr[followup])
                golds_for_eval.append(label)

        try:
            auc = metrics.average_precision_score(golds_for_eval, probs_for_eval, average="samples")
        except Exception as e:
            warnings.warn(f"Failed to calculate AUC because {e}")
            auc = np.nan  # <-- use numeric NaN instead of string
        aucs_per_year[followup] = auc

    return aucs_per_year




# --------------------------------------
# Performance acorss cancer subtypes
# ---------------------------------------

def bootstrap_auc_by_cancer_type(
    event_times,
    predictions,
    event_observed,
    cancer_categories,
    n_bootstrap=2000,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for AUC by cancer categories.

    Ensures that each year has at least one positive and one negative sample; 
    otherwise, resamples until valid.

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        cancer_categories: Array of categorical cancer type labels (N,), values in [0–6]
        n_bootstrap: Number of bootstrap samples (default=2000)
        alpha: Significance level for confidence intervals (default=0.05)

    Returns:
        auc_summary_by_cancer_type: Dict mapping cancer categories to dicts with year keys
                                    and values as (mean_auc, (lower_CI, upper_CI))
    """
    categories = list(range(7))  # 0–6
    auc_results_by_cancer = {c: {f"Year {i+1}": [] for i in range(5)} for c in categories}

    cancer_categories = np.array(cancer_categories, dtype=int)

    for cat in categories:
        cat_indices = np.where(cancer_categories == cat)[0]

        if len(cat_indices) == 0:
            warnings.warn(f"Skipping cancer category {cat}: no samples.")
            continue

        event_times_cat = event_times[cat_indices]
        predictions_cat = predictions[cat_indices]
        event_observed_cat = event_observed[cat_indices]

        cases = np.where(event_observed_cat == 1)[0]
        controls = np.where(event_observed_cat == 0)[0]
        if len(cases) == 0 or len(controls) == 0:
            warnings.warn(f"Skipping cancer category {cat}: no positive or negative cases for bootstrapping.")
            continue
        for _ in range(n_bootstrap):
            sample_cases = resample(cases, replace=True, n_samples=len(cases))
            sample_controls = resample(controls, replace=True, n_samples=len(controls))

            boot_idx = np.concatenate([sample_cases, sample_controls])

            yearly_aucs_sample = compute_auc_x_year_auc(
                predictions_cat[boot_idx], event_times_cat[boot_idx], event_observed_cat[boot_idx]
            )

            for year, auc in yearly_aucs_sample.items():
                auc_results_by_cancer[cat][f"Year {year+1}"].append(auc)

    # Compute mean and CI for each category/year
    auc_summary_by_cancer = {}
    for cat, auc_results in auc_results_by_cancer.items():
        auc_summary_by_cancer[cat] = {}
        for year, auc_values in auc_results.items():
            valid_values = np.array([v for v in auc_values if np.isfinite(v)])
            if len(valid_values) > 0:
                lower = np.percentile(valid_values, 100 * alpha / 2)
                upper = np.percentile(valid_values, 100 * (1 - alpha / 2))
                auc_summary_by_cancer[cat][year] = (np.mean(valid_values), (lower, upper))
            else:
                auc_summary_by_cancer[cat][year] = (None, (None, None))
    return auc_summary_by_cancer



def bootstrap_c_index_by_cancer_type(
    event_times,
    predictions,
    event_observed,
    cancer_categories,
    censoring_dist,
    n_bootstrap=2000,
    alpha=0.05,
        save_json_path=None,
):
    categories = np.unique(cancer_categories)
    results = {}

    for cat in categories:
        idx = np.where(cancer_categories == cat)[0]

        if len(idx) < 10 or np.sum(event_observed[idx]) < 5:
            results[cat] = (None, (None, None))
            continue

        c_vals = []

        cases = np.where(event_observed == 1)[0]
        controls = np.where(event_observed == 0)[0]
        if len(cases) == 0 or len(controls) == 0:
            warnings.warn(f"Skipping cancer category {cat}: no positive or negative cases for bootstrapping.")
            continue
        for _ in range(n_bootstrap):
            sample_cases = resample(cases, replace=True, n_samples=len(cases))
            sample_controls = resample(controls, replace=True, n_samples=len(controls))

            boot_idx = np.concatenate([sample_cases, sample_controls])

            try:
                c = concordance_index_ipcw(
                    event_times[boot_idx],
                    predictions[boot_idx],
                    event_observed[boot_idx],
                    censoring_dist,
                )
                c_vals.append(c)
            except ZeroDivisionError:
                warnings.warn(
                    f"Skipping bootstrap sample due to no admissible pairs."
                )
                continue

        c_vals = np.array(c_vals)
        lower = np.percentile(c_vals, 100 * alpha / 2)
        upper = np.percentile(c_vals, 100 * (1 - alpha / 2))

        results[cat] = (np.mean(c_vals), (lower, upper))

    return results


def auc_by_cancer_type(
    event_times,
    predictions,
    event_observed,
    cancer_categories,
):
    """
    Compute AUC by cancer categories and year (no bootstrapping).

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        cancer_categories: Array of categorical cancer type labels (N,), values in [0–6]

    Returns:
        auc_summary_by_cancer: Dict mapping cancer categories to dicts with year keys
                               and AUC values.
    """
    categories = list(range(7))  # 0–6
    auc_summary_by_cancer = {}

    cancer_categories = np.array(cancer_categories, dtype=int)

    for cat in categories:
        cat_indices = np.where(cancer_categories == cat)[0]
        if len(cat_indices) == 0:
            print(f"Skipping category {cat}: no samples.")
            continue

        event_times_cat = event_times[cat_indices]
        predictions_cat = predictions[cat_indices]
        event_observed_cat = event_observed[cat_indices]

        cancer_indices = np.where(event_observed_cat == 1)[0]
        non_cancer_indices = np.where(event_observed_cat == 0)[0]

        if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
            print(f"Skipping category {cat}: missing cancer or non-cancer cases.")
            continue

        yearly_aucs = compute_auc_x_year_auc(
            predictions_cat, event_times_cat, event_observed_cat
        )

        auc_summary_by_cancer[cat] = {f"Year {year + 1}": auc for year, auc in yearly_aucs.items()}

    return auc_summary_by_cancer


# --------------------------------------
# Performance acorss density categories
# ---------------------------------------

def compute_auc_by_density_category(predictions, event_times, event_observed, density_categories):
    """
    Compute AUC for each density category (A, B, C, D) and each follow-up year.

    Args:
        predictions: List or array of predicted probabilities (N, 5).
        event_times: List or array of event/censor times (N,).
        event_observed: List or array of event indicators (N,).
        density_categories: List or array of density categories (N,), values in {"A","B","C","D"}.

    Returns:
        aucs_by_density: Dict with keys 'A','B','C','D', each mapping to a dict of yearly AUCs.
    """
    aucs_by_density = {"A": {}, "B": {}, "C": {}, "D": {}}
    density_categories = np.array([map_density(v) for v in density_categories])

    for density in ["A", "B", "C", "D"]:
        idx = [i for i, cat in enumerate(density_categories) if cat == density]
        probs = [predictions[i] for i in idx]
        event_times_filtered = [event_times[i] for i in idx]
        event_observed_filtered = [event_observed[i] for i in idx]

        aucs_by_density[density] = compute_auc_x_year_auc(probs, event_times_filtered, event_observed_filtered)

    return aucs_by_density


def compute_c_index_by_density(event_times, predictions, event_observed, density_categories, censoring_dist):
    """
    Compute the concordance index (C-index) for each density category without bootstrapping.

    Args:
        event_times: Array of event/censoring times (N,).
        predictions: Array of predicted risk scores (N,).
        event_observed: Array of binary event indicators (N,).
        density_categories: Array of density categories (N,), values in {"A","B","C","D"}.
        censoring_dist: Censoring distribution used for IPCW calculation.

    Returns:
        c_indexes_by_density: Dict mapping density categories to their corresponding C-index values.
    """
    c_indexes_by_density = {"A": None, "B": None, "C": None, "D": None}
    density_categories = np.array([map_density(v) for v in density_categories])

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]
        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        try:
            c_index = concordance_index_ipcw(
                event_times_density,
                predictions_density,
                event_observed_density,
                censoring_dist,
            )
        except Exception as e:
            print(f"Error calculating C-index for density {density}: {e}")
            c_index = None

        c_indexes_by_density[density] = c_index

    return c_indexes_by_density


def bootstrap_c_index_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    censoring_dist,
    n_bootstrap=2000,
    alpha=0.05,
    save_json_path=None,
):
    """
    Compute bootstrap confidence intervals for concordance index by density categories,
    optionally saving the bootstrap samples to a JSON file.

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        density_categories: Array of categorical density labels (N,), values in ["A", "B", "C", "D"]
        censoring_dist: Censoring distribution for IPCW calculation
        n_bootstrap: Number of bootstrap samples (default=1000)
        alpha: Significance level for confidence intervals (default=0.05)
        save_json_path: Optional path to save bootstrap results JSON file (default=None)

    Returns:
        c_index_summary_by_density: Dict mapping density categories to
                                   (mean_c_index, (lower_CI, upper_CI)) tuples.
    """

    c_index_results_by_density = {density: [] for density in ["A", "B", "C", "D"]}
    density_categories = np.array([map_density(v) for v in density_categories])

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]
        if len(density_indices) == 0:
            continue

        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        cases = np.where(event_observed_density == 1)[0]
        controls = np.where(event_observed_density==0)[0]

        for _ in range(n_bootstrap):
            sample_cases = resample(cases, replace=True, n_samples=len(cases))
            sample_controls = resample(controls, replace=True, n_samples=len(controls))

            indices = np.concatenate([sample_cases, sample_controls])
            event_times_sample = event_times_density[indices]
            predictions_sample = predictions_density[indices]
            event_observed_sample = event_observed_density[indices]

            c_index = concordance_index_ipcw(
                event_times_sample,
                predictions_sample,
                event_observed_sample,
                censoring_dist,
            )
            c_index_results_by_density[density].append(c_index)

    if save_json_path is not None:
        filename = "mbox_plots_c_index_density_results.json"
        file_path = os.path.join(save_json_path, filename)
        c_index_serializable = {
            density: list(map(float, values))
            for density, values in c_index_results_by_density.items()
        }
        with open(file_path, "w") as f:
            json.dump(c_index_serializable, f)
        print(f"[INFO] Saved bootstrap C-index samples to {file_path}")

    c_index_summary_by_density = {}
    for density, c_index_values in c_index_results_by_density.items():
        if c_index_values:
            lower = np.percentile(c_index_values, 100 * alpha / 2)
            upper = np.percentile(c_index_values, 100 * (1 - alpha / 2))
            c_index_summary_by_density[density] = (np.mean(c_index_values), (lower, upper))
        else:
            c_index_summary_by_density[density] = (None, (None, None))

    return c_index_summary_by_density, c_index_results_by_density


def bootstrap_auc_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    n_bootstrap=2000,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for AUC by density categories
    """

    auc_results_by_density = {
        d: {f"Year {i+1}": [] for i in range(5)} for d in ["A", "B", "C", "D"]
    }

    density_categories = np.array([map_density(v) for v in density_categories])

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]

        if len(density_indices) == 0:
            print(f"Skipping density '{density}': no samples.")
            continue

        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        n_density = len(density_indices)

        cases_density = np.where(event_observed_density == 1)[0]
        controls_density = np.where(event_observed_density == 0)[0]
        if len(cases_density) == 0 or len(controls_density) == 0:
            print(f"Skipping density '{density}': insufficient class balance.")
            continue
        for _ in range(n_bootstrap):
            sample_cases = resample(cases_density, replace=True, n_samples=len(cases_density))
            sample_controls = resample(controls_density, replace=True, n_samples=len(controls_density))

            sample_indices = np.concatenate([sample_cases, sample_controls])

            event_times_sample = event_times_density[sample_indices]
            predictions_sample = predictions_density[sample_indices]
            event_observed_sample = event_observed_density[sample_indices]

            yearly_aucs_sample = compute_auc_x_year_auc(
                predictions_sample,
                event_times_sample,
                event_observed_sample,
            )

            for year, auc in yearly_aucs_sample.items():
                auc_results_by_density[density][f"Year {year + 1}"].append(auc)

    auc_summary_by_density = {}
    for density, auc_results in auc_results_by_density.items():
        auc_summary_by_density[density] = {}
        for year, auc_values in auc_results.items():
            if auc_values:
                lower = np.percentile(auc_values, 100 * alpha / 2)
                upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
                auc_summary_by_density[density][year] = (
                    np.mean(auc_values),
                    (lower, upper),
                )
            else:
                auc_summary_by_density[density][year] = (None, (None, None))

    return auc_summary_by_density


def bootstrap_c_index_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    censoring_dist,
    n_bootstrap=2000,
    alpha=0.05,
    save_json_path=None,
):
    """
    Compute bootstrap confidence intervals for concordance index by density categories
    """

    c_index_results_by_density = {d: [] for d in ["A", "B", "C", "D"]}
    density_categories = np.array([map_density(v) for v in density_categories])

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]

        if len(density_indices) == 0:
            print(f"Skipping density '{density}': no samples.")
            continue

        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]
        n_events = event_observed_density.sum()

        if n_events == 0:
            print(f"[WARNING] Skipping density '{density}': no observed events.")
            continue
        
        cases_density = np.where(event_observed_density == 1)[0]
        controls_density = np.where(event_observed_density == 0)[0]
        if len(cases_density) == 0 or len(controls_density) == 0:
            print(f"Skipping density '{density}': insufficient class balance.")
            continue
        for _ in range(n_bootstrap):
            sample_cases = resample(cases_density, replace=True, n_samples=len(cases_density))
            sample_controls = resample(controls_density, replace=True, n_samples=len(controls_density))

            indices = np.concatenate([sample_cases, sample_controls])
            event_observed_sample = event_observed_density[indices]
            event_times_sample = event_times_density[indices]
            predictions_sample = predictions_density[indices]

            try:
                c_index = concordance_index_ipcw(
                    event_times_sample,
                    predictions_sample,
                    event_observed_sample,
                    censoring_dist,
                )
                c_index_results_by_density[density].append(c_index)
            except ZeroDivisionError:
                # Should rarely happen now, but safe to catch
                warnings.warn(
                    f"Skipping bootstrap sample for density {density} due to no admissible pairs."
                )
                continue

    if save_json_path is not None:
        filename = "mbox_plots_c_index_density_results.json"
        file_path = os.path.join(save_json_path, filename)
        with open(file_path, "w") as f:
            json.dump(
                {k: list(map(float, v)) for k, v in c_index_results_by_density.items()},
                f,
            )
        print(f"[INFO] Saved bootstrap C-index samples to {file_path}")

    c_index_summary_by_density = {}
    for density, values in c_index_results_by_density.items():
        if values:
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            c_index_summary_by_density[density] = (
                np.mean(values),
                (lower, upper),
            )
        else:
            c_index_summary_by_density[density] = (None, (None, None))

    return c_index_summary_by_density, c_index_results_by_density


# --------------------------------------
# Performance acorss race categories
# ---------------------------------------

def bootstrap_auc_by_race(
    event_times,
    predictions,
    event_observed,
    race_categories,
    n_bootstrap=2000,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for AUC by race categories.
    """

    race_categories = np.array([
        r if isinstance(r, str) and r.strip() != "" else "Unknown"
        for r in race_categories
    ])

    RACES = [
        "Caucasian or White",
        "African American  or Black",
        "Asian",
        "American Indian or Alaskan Native",
        "Native Hawaiian or Other Pacific Islander",
        "Multiple",
        "Unknown",
        "Unavailable or Unreported",
    ]

    auc_results_by_race = {
        r: {f"Year {i + 1}": [] for i in range(5)} for r in RACES
    }

    for race in RACES:
        idx = np.where(race_categories == race)[0]

        if len(idx) == 0:
            print(f"[WARN] Skipping race '{race}' — no samples")
            continue

        et_r = event_times[idx]
        pred_r = predictions[idx]
        obs_r = event_observed[idx]

        cases = np.where(obs_r == 1)[0]
        controls = np.where(obs_r == 0)[0]
        if len(cases) == 0 or len(controls) == 0:
            print(f"[WARN] Skipping race '{race}' — no positive or negative samples")
            continue
        for _ in range(n_bootstrap):
            sample_cases = resample(cases, replace=True, n_samples=len(cases))
            sample_controls = resample(controls, replace=True, n_samples=len(controls))

            boot_idx = np.concatenate([sample_cases, sample_controls])

            yearly_aucs = compute_auc_x_year_auc(
                pred_r[boot_idx],
                et_r[boot_idx],
                obs_r[boot_idx],
            )

            for year, auc in yearly_aucs.items():
                auc_results_by_race[race][f"Year {year + 1}"].append(auc)

    # CI summary
    auc_summary_by_race = {}
    for race, auc_dict in auc_results_by_race.items():
        auc_summary_by_race[race] = {}
        for year, values in auc_dict.items():
            valid = np.array(
                [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            )

            if len(valid) > 0:
                low = np.percentile(valid, 100 * alpha / 2)
                high = np.percentile(valid, 100 * (1 - alpha / 2))
                auc_summary_by_race[race][year] = (np.mean(valid), (low, high))
            else:
                auc_summary_by_race[race][year] = (None, (None, None))

    return auc_summary_by_race


def bootstrap_c_index_by_race(
    event_times,
    predictions,
    event_observed,
    race_categories,
    censoring_dist,
    n_bootstrap=2000,
    alpha=0.05,
    save_json_path=None,
):
    """
       Compute bootstrap confidence intervals for C-index by race categories.
    """

    race_categories = np.array([
        r if isinstance(r, str) and r.strip() != "" else "Unknown"
        for r in race_categories
    ])

    RACES = [
        "Caucasian or White",
        "African American  or Black",
        "Asian",
        "American Indian or Alaskan Native",
        "Native Hawaiian or Other Pacific Islander",
        "Multiple",
        "Unknown",
        "Unavailable or Unreported",
    ]

    cindex_results_by_race = {r: [] for r in RACES}

    for race in RACES:
        idx = np.where(race_categories == race)[0]

        if len(idx) < 5 or np.sum(event_observed[idx]) < 2:
            print(f"[INFO] Skipping race '{race}' — insufficient samples/events")
            continue

        et_r = event_times[idx]
        pred_r = predictions[idx]
        obs_r = event_observed[idx]

        cases = np.where(obs_r == 1)[0]
        controls = np.where(obs_r == 0)[0]
        if len(cases) == 0 or len(controls) == 0:
            print(f"[WARN] Skipping race '{race}' — no positive or negative samples")
            continue
        for _ in range(n_bootstrap):
            sample_cases = resample(cases, replace=True, n_samples=len(cases))
            sample_controls = resample(controls, replace=True, n_samples=len(controls))

            boot_idx = np.concatenate([sample_cases, sample_controls])

            et_sample = et_r[boot_idx]
            pred_sample = pred_r[boot_idx]
            obs_sample = obs_r[boot_idx]

            try:
                cidx = concordance_index_ipcw(
                    et_sample,
                    pred_sample,
                    obs_sample,
                    censoring_dist,
                )
                if np.isfinite(cidx):
                    cindex_results_by_race[race].append(cidx)
            except ZeroDivisionError:
                continue  # skip this bootstrap sample


    # Optional JSON save
    if save_json_path is not None:
        fp = os.path.join(save_json_path, "c_index_race_bootstrap_samples.json")
        serializable = {r: list(map(float, v)) for r, v in cindex_results_by_race.items()}
        with open(fp, "w") as f:
            json.dump(serializable, f)
        print(f"[INFO] Saved bootstrap C-index samples to {fp}")

    # CI summary
    cindex_summary_by_race = {}
    for race, vals in cindex_results_by_race.items():
        valid = np.array(
            [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        )

        if len(valid) > 0:
            low = np.percentile(valid, 100 * alpha / 2)
            high = np.percentile(valid, 100 * (1 - alpha / 2))
            cindex_summary_by_race[race] = (np.mean(valid), (low, high))
        else:
            cindex_summary_by_race[race] = (None, (None, None))

    return cindex_summary_by_race, cindex_results_by_race

