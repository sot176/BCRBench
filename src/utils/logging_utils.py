import logging
import torch
import os 
import json

from .utils import compute_c_index_by_density, compute_auc_x_year_auc, auc_by_cancer_type


def checkpoint(model, filename):
    """
    Save model state dictionary to a file.

    Args:
        model: PyTorch model whose state_dict will be saved.
        filename: String path to save the model state dictionary.
    """
    torch.save(model.state_dict(), filename)


def create_logger(log_path):
    """
    Create a logger that writes INFO-level logs to a specified file.

    Args:
        log_path: Path to the log file.

    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    f_handler = logging.FileHandler(log_path)
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(f_handler)
    return logger


def print_results(results):
    """
    Nicely print nested dictionaries or key-value pairs.

    Args:
        results: Dict or nested dict to print.
    """
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


def save_model_results_to_file(probs, censor_times, golds, density_categories, censoring_dist, cancer_categories, out_dir):
    """
    Computes and saves the AUC values, C-index by density, predictions,
    censoring times, and event labels to a JSON file.

    Args:
        probs: Array of predicted probabilities (N, T).
        censor_times: Array of event or censoring times (N,).
        golds: Array of event indicators (1 if event occurred, else 0) (N,).
        density_categories: Array of breast density categories (N,), values in {"A", "B", "C", "D"}.
        censoring_dist: Censoring distribution used for IPCW calculation.
        out_dir: Directory path to save the output JSON file.

    Saves:
        model_results.json containing:
            - "C_index_by_density": Concordance index per density category.
            - "auc_per_year": AUC values for years 1–5.
            - "predictions": Predicted probabilities (as list).
            - "censor_times": Event/censor times (as list).
            - "golds": Ground truth labels (as list).
    """
    # Compute AUC per year for the current model
    aucs_per_year = compute_auc_x_year_auc(probs, censor_times, golds)

    # Compute C-index by density
    c_indexes_by_density = compute_c_index_by_density(
        censor_times,
        probs,
        golds,
        density_categories,
        censoring_dist,
    )
    auc_by_cancer = auc_by_cancer_type(censor_times,
    probs,
    golds,
    cancer_categories)

    # Prepare results dictionary
    results_dict = {
        "C_index_by_density": c_indexes_by_density,
        "auc_per_year": aucs_per_year,
        "auc_by_cancer_type":auc_by_cancer,
        "predictions": probs.tolist(),
        "censor_times": censor_times.tolist(),
        "golds": golds.tolist()
    }

    # Define output file path
    filename = "model_results.json"
    file_path = os.path.join(out_dir, filename)

    # Save the results to a JSON file
    with open(file_path, 'w') as file:
        json.dump(results_dict, file, indent=4)
        print(f"Results for all models saved to {file_path}")