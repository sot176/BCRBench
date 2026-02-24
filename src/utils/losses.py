import torch.nn.functional as F
import torch


def get_risk_loss_BCE( pred, y_true, y_mask):
    """
    Binary cross-entropy loss adapted for cumulative risk prediction with masking.
    Args:
        pred: Logits for cumulative risk, tensor of shape [B, T]
        y_true: Binary ground truth labels, tensor of shape [B, T]
                (1 if event happened by year t)
        y_mask: Mask tensor of shape [B, T], where 1 indicates valid data for year t
                and 0 indicates censored or invalid data

    Returns:
        masked_loss: Scalar tensor representing the masked binary cross-entropy loss.
    """

    y_mask = y_mask.to(pred.device)
    y_true = y_true.to(pred.device)
    masked_loss = F.binary_cross_entropy_with_logits(
        pred, y_true.float(), weight=y_mask.float(),  reduction='sum'
    ) / torch.sum(y_mask.float())

    return masked_loss
