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

models/
└── your_model_name/
├── model.py
└── model_utils.py

### Recommended structure:
- **`model.py`** → Main model implementation  
- **`model_utils.py`** → Helper functions, custom layers, utilities  

---

## 🧩 2. Inherit from `BaseRiskModel`

All models must inherit from the shared base class: `models/common_parts/base_models.py`



### Example:

```python
from models.common_parts.base_models import BaseRiskModel

class YourModel(BaseRiskModel):
    def __init__(self, config):
        super().__init__(config)
        # define layers

    def forward(self, batch):
        # implement forward pass
        return output
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
