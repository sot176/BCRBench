# Mammography Image Preprocessing

This document describes the preprocessing pipeline applied to all mammography images prior to model training and evaluation.

## 1. Data Format Conversion

- All mammograms were originally stored in **DICOM format**
- Converted to **16-bit grayscale PNG images**
- This conversion preserves high-intensity resolution while enabling efficient processing in deep learning pipelines

## 2. Image Preprocessing Pipeline

All images were preprocessed following the same protocol:

- **Background removal** to eliminate irrelevant black borders and artifacts  
- **Breast region segmentation** to isolate the region of interest and remove non-breast content, including **annotations and text commonly present in mammograms**  
- **Resizing** to a fixed resolution of **1664 × 2048 pixels**  
- **Intensity normalization** to standardize pixel value distributions across scans  

## 3. Dataset Split

- Data split was performed at the **patient level** to avoid data leakage  
- Splits were defined as:
  - **Training:** 50%  
  - **Validation:** 20%  
  - **Test:** 30%  

## 4. Summary

This preprocessing pipeline ensures:
- Consistent image resolution and intensity scaling  
- Removal of irrelevant background and annotation artifacts  
- Proper separation of patient data across splits  
- Compatibility with deep learning-based mammography analysis models  