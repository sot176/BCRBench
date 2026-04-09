from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
# Abstract Base Model — defines the interface all models must follow
# ══════════════════════════════════════════════════════════════════

class BaseRiskModel(nn.Module, ABC):
    """
    Abstract base class for all breast cancer risk prediction models.
    Defines the interface that training/evaluation code depends on.
    """

    @abstractmethod
    def forward(self, batch: Dict) -> Dict:
        """Run forward pass. Returns dict of outputs."""
        pass

    @abstractmethod
    def get_risk_heads(self, outputs: Dict, batch: Dict) -> Dict[str, Tuple]:
        """
        Returns dict of {head_name: (logits, target, mask)} for loss computation.
        """
        pass

    @abstractmethod
    def get_primary_risk_head(self, outputs: Dict) -> torch.Tensor:
        """Returns the main risk prediction tensor for evaluation metrics (AUC, C_index)."""
        pass

