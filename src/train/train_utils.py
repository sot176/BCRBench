import logging
from typing import Tuple, Dict, List, Optional, Any

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import (
    concordance_index_ipcw,
    get_censoring_dist,
    compute_auc_x_year_auc,
)

def train_one_epoch(
    model_risk: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    accelerator: Accelerator,
    warmup_scheduler: torch.optim.lr_scheduler.LambdaLR,
    global_step: int,
    warmup_steps: int,
    loss_fn: Any,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Execute a single training epoch with gradient accumulation and warmup.

    Performs forward pass, computes loss, and updates model parameters. Handles distributed
    training via Accelerator and updates learning rate using warmup scheduler.

    Args:
        model_risk: PyTorch model for risk prediction.
        train_loader: DataLoader for training samples.
        optimizer: Optimizer for model parameters (e.g., AdamW).
        accelerator: Hugging Face Accelerator for distributed training and mixed precision.
        warmup_scheduler: LambdaLR scheduler for learning rate warmup.
        global_step: Current global training step (used for warmup scheduling).
        warmup_steps: Number of steps for learning rate warmup.
        loss_fn: Loss function callable that takes model outputs, batch, and base model.

    Returns:
        Tuple containing:
            - avg_risk_loss (float): Average loss across all batches.
            - c_index (float): Concordance index for survival prediction (0.0 on non-main processes).
            - auc_results (Dict): Year-wise AUC scores (empty dict on non-main processes).
    """
    model_risk.train()
    running_risk_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_times: List[torch.Tensor] = []
    all_events: List[torch.Tensor] = []

    for batch in train_loader:

        outputs = model_risk(batch)

        base_model = accelerator.unwrap_model(model_risk)
        risk_loss = loss_fn(outputs, batch, base_model)

        running_risk_loss += accelerator.gather(risk_loss.detach()).mean().item()
        optimizer.zero_grad()
        accelerator.backward(risk_loss)
        optimizer.step()

        # Update warm-up scheduler
        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

        pred_risk = base_model.get_primary_risk_head(outputs)

        all_preds.append(
            accelerator.gather(pred_risk.detach())
        )

        # Gather results for metric calculation
        all_times.append(accelerator.gather(batch["event_times"]))
        all_events.append(accelerator.gather(batch["event_observed"]))

    avg_risk_loss = running_risk_loss / len(train_loader)
    c_index, auc_results = 0, {}

    accelerator.wait_for_everyone()

    # Calculate metrics on the main process
    if accelerator.is_main_process:
        preds = torch.cat(all_preds).cpu().numpy()
        times = torch.cat(all_times).cpu().numpy().astype(int)
        events = torch.cat(all_events).cpu().numpy()
        censoring_dist = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring_dist)
        auc_results = compute_auc_x_year_auc(preds, times, events)

    return avg_risk_loss, c_index, auc_results


def evaluate(
    model_risk: torch.nn.Module,
    valid_loader: DataLoader,
    accelerator: Accelerator,
    loss_fn: Any,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Evaluate model on validation set without updating model parameters.

    Computes predictions and metrics on validation data using no_grad context
    for memory efficiency in distributed setting.

    Args:
        model_risk: PyTorch model for risk prediction.
        valid_loader: DataLoader for validation samples.
        accelerator: Hugging Face Accelerator for distributed training.
        loss_fn: Loss function callable that takes model outputs, batch, and base model.

    Returns:
        Tuple containing:
            - avg_risk_loss (float): Average validation loss.
            - c_index (float): Concordance index for validation set (0.0 on non-main processes).
            - auc_results (Dict): Year-wise AUC scores (empty dict on non-main processes).
    """
    model_risk.eval()
    running_risk_loss = 0.0
    val_preds: List[torch.Tensor] = []
    val_times: List[torch.Tensor] = []
    val_events: List[torch.Tensor] = []
    
    with torch.no_grad():
        for batch_val in valid_loader:
            outputs_val = model_risk(batch_val)

            base_model = accelerator.unwrap_model(model_risk)

            risk_loss_val = loss_fn(outputs_val, batch_val, base_model)

            running_risk_loss += accelerator.gather(risk_loss_val.detach()).mean().item()

            pred_risk = base_model.get_primary_risk_head(outputs_val)

            val_preds.append(
                accelerator.gather(pred_risk.detach())
            )
            val_times.append(accelerator.gather(batch_val["event_times"]))
            val_events.append(accelerator.gather(batch_val["event_observed"]))

    avg_risk_loss = running_risk_loss / len(valid_loader)
    c_index, auc_results = 0, {}

    accelerator.wait_for_everyone()

    # Calculate metrics on the main process
    if accelerator.is_main_process:
        predictions_val = torch.cat(val_preds).cpu().numpy()
        event_times_val = torch.cat(val_times).cpu().numpy().astype(int)
        event_observed_val = torch.cat(val_events).cpu().numpy()
        censoring_dist_val = get_censoring_dist(event_times_val, event_observed_val)
        c_index = concordance_index_ipcw(event_times_val, predictions_val, event_observed_val, censoring_dist_val)
        auc_results = compute_auc_x_year_auc(predictions_val, event_times_val, event_observed_val)

    return avg_risk_loss, c_index, auc_results



def get_model_size(model: torch.nn.Module, accelerator: Accelerator) -> float:
    """
    Calculate total model size in MB, accounting for parameters and buffers.

    Unwraps the model from Accelerator wrapper and computes memory footprint
    considering both parameters and buffers. Logs result on main process.

    Args:
        model: PyTorch model (may be wrapped by Accelerator).
        accelerator: Hugging Face Accelerator instance for unwrapping.

    Returns:
        Total model size in megabytes (MB).
    """
    logger = logging.getLogger(__name__)
    unwrapped_model = accelerator.unwrap_model(model)
    param_size = sum(p.numel() * p.element_size() for p in unwrapped_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in unwrapped_model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024**2)

    if accelerator.is_main_process:
        logger.info(
            f"Model size: {total_size_mb:.2f} MB (Parameters: {param_size / 1e6:.1f}M elements)"
        )
    return total_size_mb

def linear_warmup(step: int, warmup_steps: int) -> float:
    """
    Compute learning rate multiplier for linear warmup scheduling.

    Gradually increases learning rate from 0 to 1.0 (base_lr) over warmup_steps,
    then maintains constant learning rate. Used as lambda function with PyTorch LambdaLR.

    Args:
        step: Current training step.
        warmup_steps: Total number of steps for warmup phase.

    Returns:
        Learning rate multiplier (0.0 to 1.0).
    """
    if warmup_steps == 0:
        return 1.0  # No warm-up, directly use the base learning rate
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def get_param_groups(
    args: Any,
    model: torch.nn.Module,
    base_lr: float,
    finetune_lr_scale: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with differential learning rates.

    Separates encoder parameters (with lower learning rate) from new module parameters
    (with higher learning rate) to enable fine-tuning pretrained encoders with stable learning.

    Args:
        args: Configuration arguments containing model name.
        model: PyTorch model to extract parameter groups from.
        base_lr: Base learning rate for new modules.
        finetune_lr_scale: Scaling factor for encoder learning rate. For OA-BreaCR, set to 1.0;
                          for others, defaults to 0.1.

    Returns:
        List of parameter group dicts, each containing 'params' and 'lr' keys.
    """
    encoder_params: List[torch.nn.Parameter] = []
    new_module_params: List[torch.nn.Parameter] = []

    if args.model == "OA-BreaCR":
        finetune_lr_scale = 1.0
    else:
        finetune_lr_scale = 0.1

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen params

        if "encoder" in name:
            # Any unfrozen encoder params go here
            encoder_params.append(param)
        else:
            # Everything else is a "new module"
            new_module_params.append(param)

    param_groups: List[Dict[str, Any]] = []
    if new_module_params:
        param_groups.append({"params": new_module_params, "lr": base_lr})
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": base_lr * finetune_lr_scale}
        )

    return param_groups


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    warmup_scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_c_index: float,
    path: str,
) -> None:
    """
    Save training checkpoint with model state, optimizer state, and training metadata.

    Saves unwrapped model weights and scheduler states for resuming training at exact state.
    Should only be called from main process to avoid overwriting conflicts.

    Args:
        accelerator: Hugging Face Accelerator instance for unwrapping distributed model.
        model: PyTorch model (may be wrapped by Accelerator).
        optimizer: PyTorch optimizer instance.
        scheduler: Optional ReduceLROnPlateau scheduler or None.
        warmup_scheduler: LambdaLR warmup scheduler.
        epoch: Current training epoch.
        global_step: Current global training step.
        best_c_index: Best validation C-index achieved so far.
        path: File path to save checkpoint.

    Returns:
        None. Saves checkpoint to disk.
    """
    accelerator.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_c_index": best_c_index,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "warmup_scheduler": warmup_scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    warmup_scheduler: torch.optim.lr_scheduler.LambdaLR,
    accelerator: Accelerator,
) -> Tuple[int, int, float]:
    """
    Load training checkpoint and restore model, optimizer, and scheduler states.

    Restores complete training state from checkpoint, allowing resumption from exact
    epoch and training step. Handles None schedulers gracefully.

    Args:
        checkpoint_path: Path to checkpoint file (typically .pth format).
        model: PyTorch model to load state into (will be unwrapped by Accelerator).
        optimizer: PyTorch optimizer to restore state.
        scheduler: Optional ReduceLROnPlateau scheduler or None.
        warmup_scheduler: LambdaLR warmup scheduler to restore state.
        accelerator: Hugging Face Accelerator for unwrapping distributed model.

    Returns:
        Tuple containing:
            - next_epoch (int): Next epoch to resume from (checkpoint epoch + 1).
            - global_step (int): Global training step at checkpoint.
            - best_c_index (float): Best validation C-index at checkpoint.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        RuntimeError: If state_dict keys don't match model structure.
    """
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

