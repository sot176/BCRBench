import torch
import numpy as np
import os
from tqdm import tqdm
from accelerate import Accelerator

from .test_utils import load_model, gather_tensor
from utils import (
    create_logger,
    save_model_results_to_file,
    bootstrap_c_index,
    bootstrap_auc,
    bootstrap_auc_by_density,
    bootstrap_c_index_by_density,
    bootstrap_auc_by_cancer_type,
    bootstrap_c_index_by_cancer_type,
    bootstrap_auc_by_race,
    bootstrap_c_index_by_race,
    get_censoring_dist,
    ID_TO_RACE,
)


def test_risk(
    args,
    test_loader,
    path_model,
    accelerator: Accelerator,
):
    """Evaluate trained model on test dataset."""

    is_main = accelerator.is_main_process

    logg_filename = f"test_risk_prediction_training_id_{args.id_training}.log"
    path_logger = os.path.join(args.path_test_folder, logg_filename)
    logger = create_logger(path_logger) if is_main else None

    if is_main:
        print("[INFO] Loading model...")

    model = load_model(args, path_model)

    # Prepare with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)
    base_model = accelerator.unwrap_model(model)

    # -------------------------
    # Inference loop
    # -------------------------
    if is_main:
        print("[INFO] Running inference...")

    results = {
        "preds": [],
        "times": [],
        "events": [],
        "densities": [],
        "cancers": [],
        "races": [],
    }

    with torch.inference_mode():
        progress_bar = tqdm(test_loader, disable=not is_main)

        for batch in progress_bar:
            outputs = model(batch)
            preds = base_model.get_primary_risk_head(outputs)

            results["preds"].append(gather_tensor(accelerator, preds.detach()))
            results["times"].append(gather_tensor(accelerator, batch["event_times"]))
            results["events"].append(gather_tensor(accelerator, batch["event_observed"]))
            results["densities"].append(gather_tensor(accelerator, batch["density"]))
            results["cancers"].append(gather_tensor(accelerator, batch["cancer_type"]))
            if args.dataset == "EMBED":
                results["races"].append(gather_tensor(accelerator, batch["race"]))

    # -------------------------
    # Aggregation
    # -------------------------
    if not is_main:
        return

    print("[INFO] Aggregating results...")

    predictions = torch.cat(results["preds"]).numpy()
    event_times = torch.cat(results["times"]).numpy().astype(int)
    event_observed = torch.cat(results["events"]).numpy()
    density_categories = torch.cat(results["densities"]).numpy()
    cancer_categories = torch.cat(results["cancers"]).numpy()
    if args.dataset == "EMBED":
        race_ids = torch.cat(results["races"]).numpy()
        race_categories = [ID_TO_RACE[int(r)] for r in race_ids]
    else:
        race_categories = None

    censoring_dist = get_censoring_dist(event_times, event_observed)

    save_model_results_to_file(
        predictions,
        event_times,
        event_observed,
        density_categories,
        censoring_dist,
        cancer_categories,
        args.path_test_folder,
    )

    # -------------------------
    # Metrics
    # -------------------------
    print("[INFO] Computing metrics...")

    mean_c_index, c_index_ci, c_index_scores = bootstrap_c_index(
        event_times, predictions, event_observed, censoring_dist
    )
    np.save(os.path.join(args.path_test_folder, "cindex_scores.npy"), c_index_scores)

    auc_summary, auc_arrays = bootstrap_auc(event_times, predictions, event_observed)
    np.savez(os.path.join(args.path_test_folder, "auc_scores.npz"), **auc_arrays)

    auc_by_density = bootstrap_auc_by_density(
        event_times, predictions, event_observed, density_categories
    )
    c_index_by_density, _ = bootstrap_c_index_by_density(
        event_times,
        predictions,
        event_observed,
        density_categories,
        censoring_dist,
        save_json_path=args.path_test_folder,
    )

    auc_by_cancer = bootstrap_auc_by_cancer_type(
        event_times, predictions, event_observed, cancer_categories
    )
    c_index_by_cancer = bootstrap_c_index_by_cancer_type(
        event_times,
        predictions,
        event_observed,
        cancer_categories,
        censoring_dist,
        save_json_path=args.path_test_folder,
    )

    if race_categories is not None:
        auc_by_race = bootstrap_auc_by_race(
            event_times, predictions, event_observed, race_categories
        )
        c_index_by_race, _ = bootstrap_c_index_by_race(
            event_times,
            predictions,
            event_observed,
            race_categories,
            censoring_dist,
            save_json_path=args.path_test_folder,
        )
    else:
        auc_by_race, c_index_by_race = None, None

    # -------------------------
    # Final formatting
    # -------------------------
    auc_formatted = {
        str(year): {"Mean": mean, "95% CI": ci}
        for year, (mean, ci) in auc_summary.items()
    }

    final_results = {
        "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
        "Yearly AUCs": auc_formatted,
        "AUC by density": auc_by_density,
        "C-index by density": c_index_by_density,
        "AUC by cancer type": auc_by_cancer,
        "C-index by cancer type": c_index_by_cancer,
        "AUC by race": auc_by_race,
        "C-index by race": c_index_by_race,
    }

    print("\nFinal Test Results:")
    print(final_results)

    if logger:
        logger.info(f"Final Test Results: {final_results}")