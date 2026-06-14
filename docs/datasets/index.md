# Datasets Overview

This repository currently supports two large, publicly available mammography datasets for breast cancer risk prediction:

- **Emory Breast Imaging Dataset (EMBED)**  
  https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview

- **Cohort of Screen-Aged Women Case Control (CSAW-CC)**  
  https://snd.se/en/catalogue/dataset/2021-204-1

These datasets provide complementary information and enable benchmarking of models across different populations and data characteristics.

---

## 📚 References

```bibtex
@article{embed,
  author = {Jeong, Jiwoong J. and Vey, Brianna L. and Bhimireddy, Ananth and Kim, Thomas and Santos, Thiago and Correa, Ramon and Dutt, Raman and Mosunjac, Marina and Oprea-Ilies, Gabriela and Smith, Geoffrey and Woo, Minjae and McAdams, Christopher R. and Newell, Mary S. and Banerjee, Imon and Gichoya, Judy and Trivedi, Hari},
  title = {The EMory BrEast imaging Dataset (EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic Mammographic Images},
  journal = {Radiology: Artificial Intelligence},
  volume = {5},
  number = {1},
  year = {2023}
}

@article{csawcc,
  title = {A multi-million mammography image dataset and population-based screening cohort for the training and evaluation of deep neural networks—the cohort of screen-aged women (CSAW)},
  author = {Dembrower, Karin and Lindholm, Peter and Strand, Fredrik},
  journal = {Journal of Digital Imaging},
  volume = {33},
  number = {2},
  year = {2020}
}
```

---

## 📊 Dataset Characteristics

Both datasets include:

- Multi-view mammography images (e.g., CC and MLO views)  
- Longitudinal screening exams per patient  
- Clinical metadata and annotations  
- Information for deriving breast cancer risk labels  

However, they differ in:

- Label definitions and availability  
- Data formats and preprocessing requirements  

---

## 🔧 Usage in This Framework

Within this repository, both datasets are:

- Standardized into a unified CSV format  
- Processed to generate time-to-cancer labels  
- Used for training and evaluating risk prediction models  

---

## 📚 Further Documentation

For detailed instructions on how to use these datasets, please refer to:

- **Preprocessing** → Data cleaning, filtering, and preparation  
- **CSV File Format** → Required structure for model input  

These sections provide step-by-step guidance for preparing datasets for use within the benchmark.

---

## ⚠️ Access and Licensing

Both datasets are publicly available but may require:

- Registration or data use agreements  
- Compliance with dataset-specific licensing terms  

Please consult the official dataset pages for access requirements.

---

## 🚀 Summary

This framework enables consistent benchmarking across EMBED and CSAW-CC by:

- Harmonizing data formats  
- Standardizing label generation  
- Supporting multi-view and longitudinal analysis  

This allows fair and reproducible comparison of breast cancer risk prediction models.