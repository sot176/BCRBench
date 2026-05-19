# Breast Cancer Risk Prediction Models

📖 **Documentation:**  
https://sot176.github.io/BreastCancerRiskBenchmark/

---

Unified repository implementing state-of-the-art image-based breast cancer risk prediction models. This project aims to simplify training and evaluation of multiple models using a single, unified pipeline, supporting both single- and multi-timepoint mammogram analysis. 

We aim to encourage further research by providing a single, unified repository where all image-based breast cancer risk prediction models can be collected, compared, and extended. By centralizing these implementations, we hope to accelerate progress in the field and make it easier for researchers to benchmark and build upon existing methods.

---

## Table of Contents
1. 📘 [Overview](#overview)  
2. ⚙️ [Models Implemented](#models-implemented)  
3. 🗂️ [Datasets](#datasets)
4. ▶️ [Usage](#usage)  
5. 📄 [Citation](#citation) 
6. 📝 [License](#license) 

---

## Overview
This repository provides implementations of several recent breast cancer risk prediction models, allowing users to:

- Train models end-to-end on mammogram datasets.

- Evaluate models with unified metrics.

- Experiment with single-view, multi-view, and longitudinal mammogram data.

---

## Models Implemented

| Model            | Conference/Journal      | Input Views & Timepoints                     | Key Idea                                                           |
| ---------------- | --------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| **[LMV-Net]()**      | submitted to MICCAI 2026  | Both views of 1 breast, 2 timepoints         | Multi-view longitudinal model with dual-stream attention leveraging both views of one breast. |
| **[ImgFeatAlign](https://doi.org/10.1007/978-3-032-04937-7_47)** | MICCAI 2025             | 1 view of 1 breast                           | Uses image-based deformation (MammoRegNet) applied in feature space for improved longitudinal comparison.   |
| **[VMRA-MaR](https://doi.org/10.1007/978-3-032-05182-0_64)**     | MICCAI 2025             | 4 screening mammograms (multiple timepoints) | Extends Mirai to longitudinal mammograms using Spatial Asymmetry Detector and Longitudinal Asymmetry Tracker.                     |
| **[OA-BReaCR](https://doi.org/10.1007/978-3-031-72378-0_15)**    | MICCAI 2024             | 1 view of 1 breast, 2 timepoints             | Learns longitudinal changes using feature-based deformation fields for better temporal alignment.             |
| **[Mirai](https://doi.org/10.1126/scitranslmed.aba4373)**        | Sci. Transl. Med. 2021         | 4 screening mammograms (single timepoint)    | Learns breast cancer risk from all four views in a single visit         |


Each model is implemented with a consistent API and unified training/evaluation pipeline.

---



## Datasets
This repository integrates two large, publicly available mammography datasets:
- **Emory Breast Imaging Dataset (EMBED)**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview}
- **Cohort of Screen-Aged Women Case Control (CSAW-CC)**: https://snd.se/en/catalogue/dataset/2021-204-1



---

## Usage
For using this framework follow the instructions below:

### 1) Clone & Setup Environment
Clone the repository, create a conda environment, and install the requirements:

```bash
git clone https://github.com/sot176/BreastCancerRiskBenchmark.git
cd BreastCancerRiskBenchmark
conda create -n bc_risk python=3.12
conda activate bc_risk
pip install -r requirements.txt
``` 

**Important**: For each script in the `scripts` folder, make sure to update the paths to point to your datasets and choose the output directories for results.

### 2) Requirements
All Python dependencies are listed in `requirements.txt`. Install them using the commands above.
 

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

---


## Citation

If you use these implementations in your research, please cite the original papers for the models and this repository.

### Repository 
```bibtex
@software{Thrun_BreastCancerRiskBenchmark_2026,
author = {Thrun, Solveig and Wetzer, Elisabeth and Wickstrøm, Kristoffer and Jenssen, Robert and Kampffmeyer, Michael},
license = {MIT},
month = may,
title = {{BreastCancerRiskBenchmark}},
url = {https://github.com/sot176/BreastCancerRiskBenchmark},
version = {0.1.0},
year = {2026}
}
```

### Model Papers
- **LMV-Net**: 
```bibtex
@inproceedings{lmv_net_2026,
author = {Thrun, Solveig and Sun, Zijun and  Salahuddin, Suaiba A. and Wickstrøm, Kristoffer and Wetzer, Elisabeth and Hansen, Stine, and Jenssen, Robert and Kampffmeyer, Michael},
title={Longitudinal Multi-View Modeling for Breast Cancer Risk Prediction},
booktitle={Under review for MICCAI 2026},
year={2026}
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

---

## License
This repository is released under the [MIT License](LICENSE).