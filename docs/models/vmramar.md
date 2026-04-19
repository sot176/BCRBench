# VMRA-MaR

 
 
## 📌 Overview

VMRAMaR is a deep learning architecture for breast cancer risk prediction that combines:

- **Patient-vise mammography encoding (e.g., CC / MLO of both breasts)**
- **Longitudinal temporal modeling via VMRNN**
- **Optional asymmetry analysis between left/right breasts**

The model produces **multi-year risk predictions** using a cumulative probability framework, integrating both temporal progression and spatial asymmetry cues.

---

## 🧠 Key Idea

VMRAMaR is built on three central principles:

- **Temporal modeling with memory:** Sequential mammography visits are modeled using a **VMRNN**, capturing disease progression over time  
- **Cross-view aggregation:** Multiple views (e.g., CC and MLO) are fused using **multi-head self-attention**  
- **Asymmetry-aware learning (optional):** Differences between left and right breast representations are explicitly modeled using spatial and longitudinal asymmetry modules  

This enables the model to capture **temporal trends**, **view consistency**, and **biological asymmetry signals**.

---

## 🏗️ Architecture


<p align="center">
  <img src="../figs/Vmramar.png" alt="VMRA-MaR Architecture" width="700"/>
</p>

The model consists of five main components:

---

### 1. Image Encoder

- Uses a pretrained CNN encoder (e.g., custom ResNet or snapshot model)  
- Pooling and classification heads are replaced with identity layers to extract feature maps  
- Optionally **frozen during training** for stability  

**Output:**
- Feature maps per image: `[B, T, V, C, H, W]`

---

### 2. View Aggregation (Multi-View Fusion)

- Spatially pooled features → `[B, T, V, C]`  
- Linear projection into embedding space  
- **Multi-head attention** applied across views (e.g., CC, MLO)  

**Output:**
- Visit-level embeddings: `[B, T, D]`

---

### 3. Temporal Modeling (VMRNN Block)

- Sequential visit embeddings processed using **VMRNN**  
- Captures temporal dependencies across screening history  
- Mean pooling over time followed by projection  

**Output:**
- Temporal feature vector: `[B, D]`

---

### 4. Asymmetry Branch (Optional)

Activated when `use_asymmetry = True`.

#### a. Spatial Asymmetry Detector (SAD)
- Compares left vs right breast feature maps  
- Produces:
  - Asymmetry heatmaps `[B, T, H_lat, W_lat]`  
  - Asymmetry coordinates  

#### b. Feature Projection
- Heatmaps flattened and projected to 512-D space  

#### c. Longitudinal Asymmetry Tracker (LAT)
- Aggregates asymmetry features across time  
- Incorporates spatial coordinates and temporal evolution  

**Output:**
- Asymmetry feature vector: `[B, 512]`

---

### 5. Fusion & Risk Prediction

- Concatenates:
  - Temporal features  
  - (Optional) asymmetry features  
- Layer normalization applied  
- Passed into **Cumulative Probability Layer (AHL)**  

**Output:**
- Risk logits: `[B, max_followup]`

---

## 🔄 Input / Output

### Input

The model expects a batch dictionary with:

- `images`: Mammography tensor  
  Shape: `[B, T, V, C, H, W]`  
  - `B`: Batch size  
  - `T`: Number of timepoints (visits)  
  - `V`: Number of views (e.g., CC, MLO)  
  - `C, H, W`: Image channels and dimensions  

- `target`: Risk labels `[B, num_years]`  
- `y_mask`: Valid label mask `[B, num_years]`  

---

### Output

The `forward` method returns:

- `logit`: Risk logits `[B, max_followup]`

---

### Helper Methods

#### **`get_risk_heads(outputs, batch)`**
Returns: `("logit_output", (logits, target, mask))`

Used for training with survival loss.

---

#### **`get_primary_risk_head(outputs)`**
Returns: `sigmoid(logit)`

This is the **final risk prediction** used for evaluation.

---

## 🧩 Integration in Framework

VMRAMaR extends `BaseRiskModel` and:

- Uses a shared batch interface for longitudinal multi-view data  
- Supports optional asymmetry modeling  
- Produces a single primary risk head  
- Integrates seamlessly with survival loss pipelines  

---

## ⚙️ Key Components

- **Image Encoder**: Pretrained CNN backbone (optionally frozen)  
- **Multihead Attention**: Cross-view feature fusion  
- **VMRNN**: Temporal sequence modeling  
- **SpatialAsymmetryDetector (SAD)**: Extracts asymmetry heatmaps  
- **LongitudinalAsymmetryTracker (LAT)**: Tracks asymmetry over time  
- **CumulativeProbabilityLayer**: Outputs time-dependent risk  

---

## 📉 Risk Prediction

- Risk is modeled as a **cumulative probability distribution over future time horizons**  
- The model outputs logits which are transformed via sigmoid for probability interpretation  

