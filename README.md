# Breast Cancer Risk Prediction Models

Unified repository implementing state-of-the-art image-based breast cancer risk prediction models. This project aims to simplify training and evaluation of multiple models using a single, unified pipeline, supporting both single- and multi-timepoint mammogram analysis. 

We aim to encourage further research by providing a single, unified repository where all image-based breast cancer risk prediction models can be collected, compared, and extended. By centralizing these implementations, we hope to accelerate progress in the field and make it easier for researchers to benchmark and build upon existing methods.

## Table of Contents
1. 📘 [Overview](#overview)  
2. ⚙️ [Models Implemented](#models-implemented)  
3. 📚 [Comparison of Models](#comparison-of-models)  
4. 🔍 [Dataset Format](#dataset-format)  
5. 💻 [Installation](#installation)
6. ▶️ [Usage](#usage)  
7. 📄 [Citation](#citation) 
8. 📝 [License](#license) 

## Overview
This repository provides implementations of several recent breast cancer risk prediction models, allowing users to:

- Train models end-to-end on mammogram datasets.

- Evaluate models with unified metrics.

- Experiment with single-view, multi-view, and longitudinal mammogram data.

## Models Implemented

| Model            | Conference/Journal      | Input Views & Timepoints                     | Key Idea                                                           |
| ---------------- | --------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| **[Mirai](https://doi.org/10.1126/scitranslmed.aba4373)**        | Sci. Transl. Med.          | 4 screening mammograms (single timepoint)    | Learns risk using all 4 views from one visit.                      |
| **[VMRA-MAR](https://doi.org/10.1007/978-3-032-05182-0_64)**     | MICCAI 2025             | 4 screening mammograms (multiple timepoints) | Extends Mirai to longitudinal mammograms.                          |
| **[OA-BReaCR](https://doi.org/10.1007/978-3-031-72378-0_15)**    | MICCAI 2024             | 1 view of 1 breast, 2 timepoints             | Focuses on single-view longitudinal features.                      |
| **[ImgFeatAlign](https://doi.org/10.1007/978-3-032-04937-7_47)** | MICCAI 2025             | 1 view of 1 breast                           | Explicit feature alignment for better longitudinal comparison.     |
| **[LMV-Net]()**      | submitted to MICCAI 2026  | Both views of 1 breast, 2 timepoints         | Multi-view longitudinal model leveraging both views of one breast. |


Each model is implemented with a consistent API and unified training/evaluation pipeline.


## Comparison of Models
 
- Single-view vs Multi-view: OA-BReaCR, ImgFeatAlign (single-view); LMV-Net (multi-view)

- Single-timepoint vs Multi-timepoint: Mirai (single); VMRA-MAR, OA-BReaCR, ImgFeatAlign, LMV-Net (multi)

- Patient-wise vs Breast-wise aggregation: Mirai, VMRA-MAR (patient-wise, all 4 images of a patient); LMV-Net (breast-wise, per breast)

## Dataset Format
 

##  Installation
For reproducing the results follow the instructions below:

## Usage
Clone the repository, create a conda environment, and install the requirements:

### 1) Clone & Setup Environment
```bash
git clone https://github.com/yourusername/breast-cancer-risk-prediction.git
cd breast-cancer-risk-prediction
conda create -n bc_risk python=3.12
conda activate bc_risk
pip install -r requirements.txt
``` 

**Important**: For each script in the `scripts` folder, make sure to update the paths to point to your datasets and choose the output directories for results.

### 2) Requirements
All Python dependencies are listed in `requirements.txt`. Install them using the commands above.e

### 3) Pre-processing of the datasets

Before training, datasets need to be preprocessed. The `preprocessing` folder contains scripts for image preparation and dataset splitting.

- **EMBED dataset:**
```bash 
preprocessing/preprocess_img_embed.py
```

- **CSAW-CC dataset:**
```bash 
preprocessing/preprocess_img_csaw_cc.py
``` 

- **Splitting datasets into train/val/test sets:**
```bash 
preprocessing/split_data.py
``` 


- **Creating dataset CSVs:**
Use the notebooks in the `notebooks` folder to generate CSV files describing your dataset, including file paths, labels, and metadata.
 

### 3) Training 
Each model has its own training script under `scripts/`.  
Run the model of your choice:

```bash
 scripts/train_mirai.sh
 scripts/train_vmra_mar.sh
 scripts/train_oa_breacr.sh
 scripts/train_imgfeatalign.sh
 scripts/train_lmv_net.sh
``` 

### 4) Evaluation
Similarly, each model has its own evaluation script:
```bash
 scripts/test_mirai.sh
 scripts/test_vmra_mar.sh
 scripts/test_oa_breacr.sh
 scripts/test_imgfeatalign.sh
 scripts/test_lmv_net.sh
``` 

 
## Citation

If you use these implementations in your research, please cite the original papers for the models **and** this repository.

### Model Papers

- **Mirai**  
```bibtex
@article{mirai_2021,
author = {Adam Yala  and Peter G. Mikhael  and Fredrik Strand  and Gigin Lin  and Kevin Smith  and Yung-Liang Wan  and Leslie Lamb  and Kevin Hughes  and Constance Lehman  and Regina Barzilay },
title = {Toward robust mammography-based models for breast cancer risk},
journal = {Science Translational Medicine},
year={2021},
volume={9},
doi={doi.org/10.1126/scitranslmed.aba4373}
}
```

- **VMRA-MaR**: 
```bibtex
@inproceedings{vmra_mar_2025,
author = {Sun, Zijun and Thrun, Solveig and Kampffmeyer, Michael},
title = {VMRA-MaR: An Asymmetry-Aware Temporal Framework for Longitudinal Breast Cancer Risk Prediction},
booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
year = {2025},
doi = {https://doi.org/10.1007/978-3-032-05182-0_64}
```

- **OA-BReaCR**: 
```bibtex
@inproceedings{oa_breacr_2024,
author = {Wang, Xin and Tan, Tao and Gao, Yuan and Marcus, Eric and Han, Luyi and Portaluri, Antonio and Zhang, Tianyu and Lu, Chunyao and Liang, Xinglong and Beets-Tan, Regina and Teuwen, Jonas and Mann, Ritse},
title = {Ordinal Learning: Longitudinal Attention Alignment Model for Predicting Time to Future Breast Cancer Events from Mammograms},
booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},  
year = {2024},
doi = {https://doi.org/10.1007/978-3-031-72378-0_15}
}
```

- **ImgFeatAlign**: 
```bibtex
@inproceedings{imgfeatalign_2025,
author = {Thrun, Solveig and Hansen, Stine and Sun, Zijun and Blum, Nele and Salahuddin, Suaiba A. and Wickstrøm, Kristoffer and Wetzer, Elisabeth and Jenssen, Robert and Stille, Maik and Kampffmeyer, Michael},
title = {Reconsidering Explicit Longitudinal Mammography Alignment for Enhanced Breast Cancer Risk Prediction},
booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
year = {2025},
doi = {10.1007/978-3-032-04937-7_47}
}
```

- **LMV-Net**: 
```bibtex
@inproceedings{lmv_net_2026,
author = {Thrun, Solveig and Sun, Zijun and  Salahuddin, Suaiba A. and Wickstrøm, Kristoffer and Wetzer, Elisabeth and Hansen, Stine, and Jenssen, Robert and Kampffmeyer, Michael},
title={Longitudinal Multi-View Modeling for Breast Cancer Risk Prediction},
booktitle={Under review for MICCAI 2026},
year={2026}
}
```

And please cite this repository:
```bibtex
@misc{breast_cancer_risk_github,
  title = {BreastCancerRiskBenchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sot176/BreastCancerRiskBenchmark}}
}
```


## License
This repository is released under the [MIT License](LICENSE).