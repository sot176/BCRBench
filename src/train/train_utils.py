import torch
from utils import (
    concordance_index_ipcw,
    get_censoring_dist,
    get_risk_loss_BCE,
    compute_auc_x_year_auc,
)

def train_one_epoch(model_risk, train_loader, optimizer, accelerator,  warmup_scheduler, global_step, warmup_steps):
    """
    Handles the training logic for a single epoch.

    Returns:
        A tuple containing (average_risk_loss, c_index) for the epoch.
    """
    model_risk.train()
    running_risk_loss = 0.0
    all_preds, all_times, all_events = [], [], []

    for batch in train_loader:

        outputs = model_risk(batch)

        base_model = accelerator.unwrap_model(model_risk)

        risk_heads = base_model.get_risk_heads(outputs, batch)

        risk_loss = sum(
            get_risk_loss_BCE(logits, target, mask)
            for logits, target, mask in risk_heads.values()
        )

        running_risk_loss += risk_loss.item()
        optimizer.zero_grad()
        accelerator.backward(risk_loss)
        optimizer.step()


        # Update warm-up scheduler
        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

        primary_logits = base_model.get_primary_risk_head(outputs)

        all_preds.append(
            accelerator.gather(torch.sigmoid(primary_logits).detach())
        )
        # Gather results for metric calculation
        all_times.append(accelerator.gather(batch["event_times"]))
        all_events.append(accelerator.gather(batch["event_observed"]))

    avg_risk_loss = running_risk_loss / len(train_loader)
    c_index, auc_results = 0, {}
    # Calculate metrics on the main process
    if accelerator.is_main_process:
        preds = torch.cat(all_preds).cpu().numpy()
        times = torch.cat(all_times).cpu().numpy().astype(int)
        events = torch.cat(all_events).cpu().numpy()
        censoring_dist = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring_dist)
        auc_results = compute_auc_x_year_auc(preds, times, events)

    return avg_risk_loss, c_index, auc_results


def evaluate(model_risk, valid_loader, accelerator):
    """
    Handles the evaluation logic for a single epoch.

    Returns:
        A tuple containing (average_risk_loss, c_index, auc_results).
    """
    model_risk.eval()
    running_risk_loss = 0.0
    val_preds, val_times, val_events = [], [], []

    with torch.no_grad():
        for batch_val in valid_loader:
            outputs_val = model_risk(batch_val)

            base_model = accelerator.unwrap_model(model_risk)

            risk_heads_val = base_model.get_risk_heads(outputs_val, batch_val)

            risk_loss_val = sum(
                get_risk_loss_BCE(logits, target, mask)
                for logits, target, mask in risk_heads_val.values()
            )
            running_risk_loss += risk_loss_val.item()

            primary_logits = base_model.get_primary_risk_head(outputs_val)

            val_preds.append(
                accelerator.gather(torch.sigmoid(primary_logits).detach())
            )
            val_times.append(accelerator.gather(batch_val["event_times"]))
            val_events.append(accelerator.gather(batch_val["event_observed"]))

    avg_risk_loss = running_risk_loss / len(valid_loader)
    c_index, auc_results = 0, {}
    # Calculate metrics on the main process
    if accelerator.is_main_process:
        predictions_val = torch.cat(val_preds).cpu().numpy()
        event_times_val = torch.cat(val_times).cpu().numpy().astype(int)
        event_observed_val = torch.cat(val_events).cpu().numpy()
        censoring_dist_val = get_censoring_dist(event_times_val, event_observed_val)
        c_index = concordance_index_ipcw(event_times_val, predictions_val, event_observed_val, censoring_dist_val)
        auc_results = compute_auc_x_year_auc(predictions_val, event_times_val, event_observed_val)

    return avg_risk_loss, c_index, auc_results



def get_model_size(model, accelerator):
    """Calculates and logs the model size, handling unwrapping from Accelerator."""
    unwrapped_model = accelerator.unwrap_model(model)
    param_size = sum(p.numel() * p.element_size() for p in unwrapped_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in unwrapped_model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    if accelerator.is_main_process:
        print(f"Model size: {total_size_mb:.2f} MB (Parameters: {param_size / 1e6:.1f}M elements)")
    return total_size_mb

def linear_warmup(step, warmup_steps):
    """
    Linear warm-up function.
    Gradually increases the learning rate from 0 to base_lr over warmup_steps.
    """
    if warmup_steps == 0:
        return 1.0  # No warm-up, directly use the base learning rate
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def get_param_groups(model, base_lr, finetune_lr_scale=0.1):
    encoder_params = []
    new_module_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen params

        if "encoder" in name:
            # Any unfrozen encoder params go here
            encoder_params.append(param)
        else:
            # Everything else is a "new module"
            new_module_params.append(param)

    param_groups = []
    if new_module_params:
        param_groups.append({"params": new_module_params, "lr": base_lr})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": base_lr * finetune_lr_scale})

    return param_groups


def save_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    warmup_scheduler,
    epoch,
    global_step,
    best_c_index,
    path
):
    accelerator.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_c_index": best_c_index,
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "warmup_scheduler": warmup_scheduler.state_dict(),
    }, path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    warmup_scheduler,
    accelerator
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])

    return (
        ckpt["epoch"] + 1,
        ckpt["global_step"],
        ckpt["best_c_index"],
    )

