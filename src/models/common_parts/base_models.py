from abc import ABC, abstractmethod
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
# Abstract Base Model — defines the interface all models must follow
# ══════════════════════════════════════════════════════════════════
class BaseRiskModel(nn.Module, ABC):
    """
    Abstract base class for all breast cancer risk prediction models.
    Defines the interface that training/evaluation code depends on.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args   

    @abstractmethod
    def forward(self, batch):
        """Run forward pass. Returns dict of outputs."""
        pass

    @abstractmethod
    def get_risk_heads(self, outputs, batch):
        """
        Returns dict of {head_name: (logits, target, mask)} for loss computation.
        """
        pass

    @abstractmethod
    def get_primary_risk_head(self, outputs):
        """Returns the main risk prediction tensor for evaluation metrics (AUC, C_index)."""
        pass