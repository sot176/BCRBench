import os
import torch
import wandb
from accelerate import Accelerator

from models.MammoRegNet import MammoRegNet
from utils import create_logger, bootstrap_auc, bootstrap_c_index

from models.model_factory import get_model
from .train_utils import train_one_epoch, evaluate, get_model_size, get_param_groups, linear_warmup, load_checkpoint, \
    save_checkpoint
from config.config import cfg


def train_val(args, train_loader, valid_loader, path_loggger, path_model, accelerator: Accelerator):
    # Initialize logger
    logger = create_logger(path_loggger) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        logger.info(f"Number Training Epochs: {args.num_epochs}")

    # --- Model and Optimizer Setup ---
    path_saved_reg_model = (cfg["paths"]["csaw_path_saved_reg_model"]
                            if args.dataset == "CSAW"
                            else cfg["paths"]["embed_path_saved_reg_model"]
                            )
    if accelerator.is_main_process: print("Path reg model:", path_saved_reg_model)

    checkpoint = torch.load(path_saved_reg_model, map_location="cpu", weights_only=True)
    new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model_reg = MammoRegNet()
    model_reg.load_state_dict(new_checkpoint)
    model_reg.eval()


    model_risk = get_model(
        args.model,
        mammo_reg_net=model_reg,
        max_followup=5,
        finetune_all=args.finetune_all,
    )
    get_model_size(model_risk, accelerator)
    total_params = sum(p.numel() for p in model_risk.parameters())
    trainable_params = sum(p.numel() for p in model_risk.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total params (M):            {total_params / 1e6:.2f} M")

    param_groups = get_param_groups(model_risk, base_lr=args.learning_rate, finetune_lr_scale=0.1)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    #  Debug print here
    name_lookup = {p: n for n, p in model_risk.named_parameters()}
    for i, group in enumerate(optimizer.param_groups):
        lr = group['lr']
        print(f"\nParam group {i}: learning rate = {lr}")
        for j, p in enumerate(group['params']):
            print(f"  {j}: {name_lookup[p]} | shape={tuple(p.shape)}")
            if j >= 4:
                print("  ...")
                break

    warmup_steps = args.warmup_steps #5000  # Number of warm-up steps
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lambda step: linear_warmup(step, warmup_steps))

    scheduler = None
    if args.use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_decay,
                                                               patience=args.patience_lr_scheduler,min_lr=1e-7, verbose=True)

        if accelerator.is_main_process:
            logger.info(f"Scheduler configured: {type(scheduler).__name__}")
            print(f"Scheduler configured: {type(scheduler).__name__}")

    # --- Prepare with Accelerator ---
    model_risk, model_reg, optimizer, train_loader, valid_loader, scheduler, warmup_scheduler  = accelerator.prepare(
        model_risk, model_reg, optimizer, train_loader, valid_loader, scheduler, warmup_scheduler
    )

    # --- WandB Initialization ---
    if accelerator.is_main_process:
        wandb.init(project="LMV_Risk_Prediction", config={
            "Optimizer": "AdamW", "architecture": "TemporalMultiViewRiskPrediction", "dataset": args.dataset,
            "epochs": args.num_epochs, "learning_rate": args.learning_rate, "Weight_decay": args.weight_decay,
        })
        wandb.define_metric("epoch", hidden=True)
        for metric in ["Training Risk Loss", "Training C-index", "Validation Risk Loss", "Validation C-index","Learning Rate",
                       "Train Year 1 AUC", "Train Year 2 AUC", "Train Year 3 AUC", "Train Year 4 AUC", "Train Year 5 AUC",  "Val Year 1 AUC", "Val Year 2 AUC", "Val Year 3 AUC", "Val Year 4 AUC", "Val Year 5 AUC"]:
            wandb.define_metric(metric, step_metric="epoch")

    # --- Training Loop ---
    start_epoch = 0
    global_step = 0
    best_c_index = 0.0
    patience_counter = 0

    if args.resume_from is not None:
        start_epoch, global_step, best_c_index = load_checkpoint(
            args.resume_from,
            model_risk,
            optimizer,
            scheduler,
            warmup_scheduler,
            accelerator,
        )
        patience_counter = 0  # reset early stopping

        if accelerator.is_main_process:
            print(f"[INFO] Resumed from {args.resume_from} at epoch {start_epoch}")

    # --------------------------------------------------
    # WandB (resume-safe)
    # --------------------------------------------------
    if accelerator.is_main_process:
        wandb.init(
            project="LMV_Risk_Prediction",
            resume="allow",
            id=args.wandb_id if hasattr(args, "wandb_id") else None,
            config=vars(args),
        )

    for epoch in range(start_epoch, args.num_epochs):
        # --- Training ---
        avg_train_loss, train_c_index, auc_results_train = train_one_epoch(args, model_risk, train_loader, optimizer, accelerator,  warmup_scheduler, global_step, warmup_steps)

        # --- Validation ---
        val_risk_loss, val_c_index, auc_results = evaluate(args, model_risk, valid_loader, accelerator)

        # --- Logging, Checkpointing, and Early Stopping (on main process) ---
        if accelerator.is_main_process:
            if accelerator.is_main_process:
                print_msg = f"##### Epoch: {epoch} ##### | Training Loss: {avg_train_loss:.4f}, Training C-index: {train_c_index:.4f} | Validation Loss: {val_risk_loss:.4f}, Validation C-index: {val_c_index:.4f}"
                logger.info(print_msg)
                print(print_msg)

            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "Training Risk Loss": avg_train_loss,
                "Training C-index": train_c_index,
                "Validation Risk Loss": val_risk_loss,
                "Validation C-index": val_c_index,
            })
            for year, auc in auc_results_train.items():
                logger.info(f"Train Year {year + 1}: AUC = {auc:.6f}")
                wandb.log({f"Train Year {year + 1} AUC": auc, "epoch": epoch})
                print(f"Train Year {year + 1}: AUC = {auc:.6f}")

            for year, auc in auc_results.items():
                logger.info(f"Val Year {year + 1}: AUC = {auc:.6f}")
                wandb.log({f"Val Year {year + 1} AUC": auc, "epoch": epoch})
                print(f"Val Year {year + 1}: AUC = {auc:.6f}")

            # Scheduler step
            if scheduler and global_step >= warmup_steps:
                scheduler.step(val_c_index)

            # Checkpoint and Early Stopping Logic
            unwrapped_model = accelerator.unwrap_model(model_risk)
            if epoch % 10 == 0:
                save_checkpoint(
                    accelerator, model_risk, optimizer, scheduler,
                    warmup_scheduler, epoch, global_step, best_c_index,
                    os.path.join(args.results_dir, f"checkpoint_{epoch:04d}.pth")
                )

            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
                save_checkpoint(
                    accelerator, model_risk, optimizer, scheduler,
                    warmup_scheduler, epoch, global_step, best_c_index,
                    os.path.join(args.results_dir,
                                 f"best_model_risk_prediction_id-{args.id_training}.pth")
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    accelerator.save(unwrapped_model.state_dict(), os.path.join(args.results_dir,
                                                                                f"early_stopping_risk_prediction_id-{args.id_training}.pth"))
                    break  # Exit loop

        # Barrier to ensure all processes stop if early stopping was triggered on the main process
        accelerator.wait_for_everyone()
        global_step += len(train_loader)

    # --- Final Cleanup ---
    if accelerator.is_main_process:
        print("[INFO] Saving final model ...")
        unwrapped_model = accelerator.unwrap_model(model_risk)
        accelerator.save(unwrapped_model.state_dict(), path_model)

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(path_model)
        wandb.log_artifact(artifact)
        wandb.finish()