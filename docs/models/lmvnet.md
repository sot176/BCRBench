# LMV-Net

## 📌 Overview

LMV-Net (Longitudinal Multi-View Network) is a deep learning model for breast cancer risk prediction from mammography. It jointly models **multi-view imaging (CC, MLO)** and **longitudinal information (current and prior exams)** to produce risk estimates over multiple future time horizons.

The model integrates temporal alignment, cross-view feature fusion, and multi-head risk prediction within a unified framework.

---

## 🧠 Key Idea

LMV-Net is built around three core ideas:

- **Longitudinal alignment:** Previous mammography feature maps are spatially aligned to current mammogrpahy feature maps using a registration network (MammoRegNet) to enable meaningful temporal comparison  
- **Cross-view fusion:** CC and MLO views are iteratively fused using self-attention and cross-attention mechanism  
- **Multi-level prediction:** Risk is predicted at both view-specific and fused multi-view levels  

This design allows the model to capture both **temporal changes** and **cross-view consistency**.

---

## 🏗️ Architecture
 
 <p align="center">
   <img src="/figs/LMV.png" alt="Image 1" width="700"/>
 </p>
 


The model consists of four main components:

### 1. Longitudinal Feature Processing
- Uses a registration network (MammoRegNet)  
- Warps prior features to align with current features
- Generates difference feature maps by subtracting the aligned prior featuers from the current features  
- Produces longitudinally aligned feature maps for CC and MLO views  

---

### 2. Cross-Attention Fusion
- A stack of cross-attention blocks  
- Iteratively exchanges information between CC and MLO features  
- Enhances view consistency and complementary feature learning  

---

### 3. Multi-View Pooling and Fusion
- Global average pooling applied to each view  
- Feature vectors concatenated and passed through a linear layer  
- Produces a fused multi-view representation  

---

### 4. Risk Prediction Heads
- Shared cumulative probability layer  
- Outputs:
  - Multi-view risk prediction  
  - CC-only risk prediction  
  - MLO-only risk prediction  

---

## 🔄 Input / Output

### Input

The model expects a batch dictionary with:

- `current_image_cc`: Current CC view `[B, 1, H, W]`  
- `previous_image_cc`: Prior CC view `[B, 1, H, W]`  
- `current_image_mlo`: Current MLO view `[B, 1, H, W]`  
- `previous_image_mlo`: Prior MLO view `[B, 1, H, W]`  
- `time_gap`: Time between exams `[B, 1]`  
- `target`: Risk labels `[B, num_years]`  
- `y_mask`: Valid label mask `[B, num_years]`  

---

### Output

The model returns:

- `risk_multi`: Multi-view fused risk logits `[B, num_years]`  
- `risk_cc`: CC-only risk logits `[B, num_years]`  
- `risk_mlo`: MLO-only risk logits `[B, num_years]`  

---

## 🧩 Integration in This Framework

LMV-Net is implemented as a subclass of `BaseRiskModel` and:

- Uses the shared data interface  
- Supports multi-view and longitudinal inputs  
- Produces multiple risk heads for multi-task training  
- Uses a unified evaluation pipeline via the primary risk head  

---

## ⚙️ Key Components

- **LongitudinalFeatureProcessor**: Handles temporal feature alignment using registration  
- **CrossAttentionBlock**: Performs cross-view feature fusion using self-attention and cross-attention  
- **CumulativeProbabilityLayer**: Outputs risk over time horizons  

---

## 📉 Risk Prediction

- Risk is modeled as **cumulative probability over time**  
- Multiple heads allow:
  - View-specific supervision  
  - Improved training stability  
