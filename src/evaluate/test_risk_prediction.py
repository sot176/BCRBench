import torch
from tqdm import tqdm
from accelerate import Accelerator
import json
import numpy as np
import os

from utils import (
    create_logger,
    save_model_results_to_file,
    bootstrap_c_index,
    bootstrap_auc_by_density,
    bootstrap_c_index_by_density,
    bootstrap_auc,
    bootstrap_c_index_by_cancer_type,
    bootstrap_auc_by_cancer_type,
    get_censoring_dist,
    bootstrap_auc_by_race, bootstrap_c_index_by_race, ID_TO_RACE
)
from config.config import cfg
from models.model_factory import get_model

 
def test_risk(
        args,
        test_loader,
        path_model,
        out_dir,
        path_logger,
        accelerator: Accelerator
):
    """
    Evaluate the trained model on the test dataset using the Accelerate framework.
    """
    # 1. Setup: Logger (only on the main process)
    logger = create_logger(path_logger) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        print("[INFO] Loading the trained models...")

    # 2. Model Loading (always on CPU first)
    # Load registration model
    # --- Model and Optimizer Setup ---
    path_saved_reg_model = (cfg["paths"]["csaw_path_saved_reg_model"]
                            if args.dataset == "CSAW"
                            else cfg["paths"]["embed_path_saved_reg_model"]
                            )
    if accelerator.is_main_process: print("Path reg model:", path_saved_reg_model)

    model_risk = get_model(
        args.model,
        args=args,
        path_saved_reg_model=path_saved_reg_model,
        max_followup=5,
        finetune_all=args.finetune_all,
    )

    checkpoint_risk = torch.load(path_model, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint_risk, dict) and "model" in checkpoint_risk:
        state_dict = checkpoint_risk["model"]
    elif isinstance(checkpoint_risk, dict) and "state_dict" in checkpoint_risk:
        state_dict = checkpoint_risk["state_dict"]
    else:
        # already a raw state_dict
        state_dict = checkpoint_risk

    # Strip DDP "module." prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_risk.load_state_dict(state_dict)
    model_risk.eval()

    # 3. Prepare models and dataloader with Accelerator
    model_risk, test_loader = accelerator.prepare(model_risk, test_loader)

    # 4. Evaluation Loop
    if accelerator.is_main_process:
        print("[INFO] Evaluating on test dataset...")

    all_preds, all_times, all_events, all_densities, all_cancers, all_races = [], [], [], [], [], []

    model_risk.eval()
    base_model = accelerator.unwrap_model(model_risk)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", disable=not accelerator.is_main_process)
        for batch in progress_bar:

            # Risk model forward
            outputs = model_risk(batch)

            primary_logits = base_model.get_primary_risk_head(outputs)

            # Gather results from all processes
            gathered_preds = accelerator.gather((torch.sigmoid(primary_logits).detach()))
            gathered_times = accelerator.gather(batch["event_times"])
            gathered_events = accelerator.gather(batch["event_observed"])
            gathered_densities = accelerator.gather(batch["density"])
            gathered_cancer_types = accelerator.gather(batch["cancer_type"])
            if args.dataset in {"EMBED"}:
                gathered_race = accelerator.gather(batch["race"])
                all_races.append(gathered_race.cpu())
            
            all_preds.append(gathered_preds.cpu())
            all_times.append(gathered_times.cpu())
            all_events.append(gathered_events.cpu())
            all_densities.append(gathered_densities.cpu())
            all_cancers.append(gathered_cancer_types.cpu())

    # 5. Metric Calculation and Logging (only on the main process)
    if accelerator.is_main_process:
        print("[INFO] Aggregating results and calculating metrics...")


        # Concatenate all gathered results
        predictions = torch.cat(all_preds).numpy()
        event_times = torch.cat(all_times).numpy().astype(int)
        event_observed = torch.cat(all_events).numpy()
        density_categories = torch.cat(all_densities).numpy()
        cancer_categories = torch.cat(all_cancers).numpy()
        if args.dataset in {"EMBED"}:
            race_ids = torch.cat(all_races).numpy()
            race_categories = [ID_TO_RACE[int(r)] for r in race_ids]

        censoring_dist = get_censoring_dist(event_times, event_observed)

        # Save predictions and labels
        save_model_results_to_file(predictions, event_times, event_observed, density_categories, censoring_dist,cancer_categories,
                                   args.path_test_folder)

        print("[INFO] Calculating metrics...")

        # C-index
        mean_c_index, c_index_ci, c_index_scores = bootstrap_c_index(event_times, predictions, event_observed, censoring_dist)
        path = os.path.join(args.path_test_folder, "cindex_scores.npy")
        np.save(path, c_index_scores)
        print("Mean C-index", mean_c_index)
        print(" C-index CI", c_index_ci)

        # Yearly AUC
        auc_summary, auc_arrays = bootstrap_auc(event_times, predictions, event_observed)
        print("AUC summary", auc_summary)
        np.savez(os.path.join(args.path_test_folder, "auc_scores.npz"), **auc_arrays)

        auc_by_density = bootstrap_auc_by_density(event_times, predictions, event_observed, density_categories)
        c_index_by_density, c_index_scores_density = bootstrap_c_index_by_density(
            event_times, predictions, event_observed, density_categories, censoring_dist, save_json_path=args.path_test_folder
        )
        print("AUC by density", auc_by_density)
        print("C index by density", c_index_by_density)
        path = os.path.join(args.path_test_folder, "cindex_scores_density.npy")
        np.save(path, c_index_scores_density)

        auc_by_cancer_types = bootstrap_auc_by_cancer_type(event_times, predictions, event_observed, cancer_categories)
        c_index_by_cancer_types = bootstrap_c_index_by_cancer_type(
            event_times, predictions, event_observed, cancer_categories, censoring_dist,
            save_json_path=args.path_test_folder
        )
        print("AUC by cancer types", auc_by_cancer_types)
        print(" C index by cancer types", c_index_by_cancer_types)

        if args.dataset in {"EMBED"}:
            auc_by_race = bootstrap_auc_by_race(event_times, predictions, event_observed, race_categories)
            c_index_by_race, c_index_scores_race = bootstrap_c_index_by_race(
                event_times, predictions, event_observed, race_categories, censoring_dist,
                save_json_path=out_dir
            )
            print("AUC by race", auc_by_race)
            print("C index by races", c_index_by_race)
        else:
            auc_by_race = None
            c_index_by_race = None
        auc_formatted = {
            f"{year}": {"Mean": mean_auc, "95% CI": ci}
            for year, (mean_auc, ci) in auc_summary.items()
        }
        print("AUC formated", auc_formatted)
        results = {
            "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
            "Yearly AUCs": auc_formatted,
            "AUC by density categories": auc_by_density,
            "C index by density categories": c_index_by_density,
            "AUC by cancer categories": auc_by_cancer_types,
            "C index by cancer categories": c_index_by_cancer_types,
            "AUC by race categories": auc_by_race,
            "C index by race categories": c_index_by_race,
        }

        # Pretty print to console
        print("Final Test Results:")
        print(results)
        logger.info(f"Final Test Results: {results}")

        # Save to JSON file
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

