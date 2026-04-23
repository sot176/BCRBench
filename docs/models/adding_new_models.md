# 🧠 Adding a New Model to the Benchmarking Framework

This repository provides a unified pipeline for training and evaluating deep learning models for risk prediction. To ensure compatibility, all new models must follow a standard structure and interface.

This guide explains how to integrate a new model into the framework.

---

## 📦 Overview

To add a new model, you need to:

1. Create a new model module  
2. Inherit from the base model class  
3. Define model-specific configuration  
4. Register the model in the factory  

---

## 📁 1. Create a Model Folder

Navigate to the `models/` directory and create a new subfolder for your model:

```
models/
└── your_model_name/
    ├── model.py
    └── model_utils.py
```

### Recommended structure:
- **`model.py`** → Main model implementation  
- **`model_utils.py`** → Helper functions, custom layers, utilities  

---

## 🧩 2. Inherit from `BaseRiskModel`

All models must inherit from the shared base class: `models/common_parts/base_models.py`


### Base Class Interface

```python
class BaseRiskModel(nn.Module, ABC):

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
        Returns dict of {head_name: (logits, target, mask)}
        used for loss computation.
        """
        pass

    @abstractmethod
    def get_primary_risk_head(self, outputs):
        """
        Returns main prediction tensor used for evaluation
        (e.g., AUC, C-index).
        """
        pass
```

### Example:

```
from models.common_parts.base_models import BaseRiskModel

class YourModel(BaseRiskModel):
    def __init__(self, args):
        super().__init__(args)
        # define layers

    def forward(self, batch):
        outputs = {}
        return outputs

    def get_risk_heads(self, outputs, batch):
        return {
            "main": (logits, target, mask)
        }

    def get_primary_risk_head(self, outputs):
        return outputs["main"]
```


## 3. ⚙️ 3. Add Model Configuration
Create a YAML configuration file for your model in:

 `config/models/your_model_name.yaml`      

**Purpose:**

Store model-specific hyperparameters that are:

- fixed by default
- but configurable by users

**Example:**

```
model_name: your_model_name

dropout: 0.3
num_heads: 4
hidden_dim: 256
num_layers: 3
```

## 🏭 4. Register the Model in model_factory.py

To make your model available in the pipeline, register it in:

`models/model_factory.py`

### Step 1: Add a builder function

```
def _build_your_model():
    from models.your_model_name.model import YourModel
    return _build_model(YourModel, args=args, **kwargs)
```

### Step 2: Add it to the registry

```
MODEL_REGISTRY = {
    "Mirai":        _build_mirai,
    "ImgFeatAlign": _build_imgfeatalign,
    "LMV-Net":      _build_lmvnet,
    "VMRA-MaR":     _build_vmramar,
    "OA-BreaCR":    _build_oa_breacr,
    "YourModel":    _build_your_model,   # ← add here
}
```

### ⚠️ Special Case: Registration-Based Models

If your model uses image registration (e.g., MammoRegNet), you must:

#### 1: Add your model name to:
```
REGISTRATION_MODELS = {"ImgFeatAlign", "LMV-Net", "YourModel"}
```

#### 2: Accept `mammo_reg_net` in your constructor:
```
def __init__(self, mammo_reg_net=None, args=None):
    super().__init__(args)
    self.mammo_reg_net = mammo_reg_net
```


## ✅ Final Checklist

Before using your model, make sure:

 Model folder created in models/
 Inherits from BaseRiskModel
 forward() implemented correctly
 YAML config added in config/models/
 Model registered in model_factory.py

## 💡 Tips
Keep your model modular and readable
Use model_utils.py for reusable components
Follow naming conventions for consistency
Test your model with a small batch before full training

## 🤝 Contributing

If you’re adding a new model for benchmarking:

Ensure it follows this structure
Provide a short description of the model (add a model page in the Documentation)
Optionally include references or related papers
