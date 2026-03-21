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
To use the models and dataset classes in this repository, the datasets must be organized as CSV files with specific columns.

---

<details>
<summary><b>Required Columns</b></summary>
<div style="font-size: 12px">

| Column Name                  | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `patient_id`                  | Unique ID for each patient                                                  |
| `exam_id`                     | Unique ID for each exam (one per timepoint)                                 |
| `ImageLateralityFinal`        | `L` for left breast, `R` for right breast                                   |
| `view`                        | Mammogram view (e.g., CC, MLO)                                             |
| `diagnosed_date_year`         | Year of breast cancer diagnosis (empty if cancer-free)                      |
| `study_date_year`             | Year the mammogram was taken                                                |
| `Time_to_Cancer_Years`       | Years until cancer diagnosis (empty if cancer-free)                         |
| `years_last_followup`         | Years from this exam until the last follow-up for the patient               |
| `density` / `libra_percentdensity` | Breast density. In EMBED, categorical (1–5). In CSAW-CC, continuous 0–100, which can be grouped into categories. |
| `path_severity` / `x_type`   | Cancer type. In EMBED, use `path_severity` (0–6). In CSAW-CC, use `x_type` (1–3). |
| `race`                        | Patient race (optional, leave empty if not available)                       |
</div>
</details>


<details>
<summary><b>Values for Cancer types</b></summary>
<div style="font-size: 12px">

| Dataset | Column | Values | Meaning |
| ------- | ------ | ------ | ------- |
| EMBED   | path_severity | 0 | Invasive cancer |
| EMBED   | path_severity | 1 | Non-invasive cancer |
| EMBED   | path_severity | 2 | High-risk lesion |
| EMBED   | path_severity | 3 | Borderline lesion |
| EMBED   | path_severity | 4 | Benign findings |
| EMBED   | path_severity | 5 | Negative (normal breast tissue) |
| EMBED   | path_severity | 6 | Non-breast cancer |
| CSAW-CC | x_type        | 1 | In situ only (non-invasive) |
| CSAW-CC | x_type        | 2 | Invasive ≤ 15 mm |
| CSAW-CC | x_type        | 3 | Invasive > 15 mm |
</div>
</details>


<details>
<summary><b>Values for Breast density</b></summary>
<div style="font-size: 12px">

| Dataset | Column | Values / Range | Meaning |
| ------- | ------ | -------------- | ------- |
| EMBED   | density | 1 | Almost entirely fat (BIRADS A) |
| EMBED   | density | 2 | Scattered fibroglandular densities (BIRADS B) |
| EMBED   | density | 3 | Heterogeneously dense (BIRADS C) |
| EMBED   | density | 4 | Extremely dense (BIRADS D) |
| EMBED   | density | 5 | Normal male |
| CSAW-CC | libra_percentdensity | 0–100 | Percentage breast density. Can be binned into categories (e.g., low, medium, high). |
</div>
</details>


<details>
<summary><b>Example: Cancer-Free Patient</b></summary>
<div style="font-size: 12px">
| patient_id | exam_id | ImageLateralityFinal | view | diagnosed_date_year | study_date_year | Time_to_Cancer_Years | years_last_followup | density | path_severity | race   |
| ---------- | ------- | ------------------ | ---- | ----------------- | --------------- | ------------------ | ----------------- | ------- | ------------- | ------ |
| 10093833      | 6488770689649000   | L                  | CC   |                   | 2015            |                    | 3                 |1       |             | Caucasian or White  |
| 10093833      | 6488770689649000   | L                  | MLO   |                   | 2015            |                    | 3                 |1       |             | Caucasian or White  |
| 10093833      | 6488770689649000   | R                  | CC   |                   | 2015            |                    | 3                 |1       |             | Caucasian or White  |
| 10093833      | 6488770689649000   | R                  | MLO   |                   | 2015            |                    | 3                 |1       |             | Caucasian or White  |
| 10093833      | 4173769076863720   | L                  | CC   |                   | 2015            |                    | 0                 |1       |             | Caucasian or White  |
| 10093833      | 4173769076863720   | L                  | MLO   |                   | 2015            |                    | 0                 |1       |             | Caucasian or White  |
| 10093833      | 4173769076863720   | R                  | CC   |                   | 2015            |                    | 0                 |1       |             | Caucasian or White  |
| 10093833      | 4173769076863720   | R                  | MLO   |                   | 2015            |                    | 0                 |1       |             | Caucasian or White  |

</div>
</details>


<details>
<summary><b>Example: Cancer Patient</b></summary>
<div style="font-size: 12px">

| patient_id | exam_id | ImageLateralityFinal | view | diagnosed_date_year | study_date_year | Time_to_Cancer_Years | years_last_followup | density | path_severity | race   |
| ---------- | ------- | ------------------ | ---- | ----------------- | --------------- | ------------------ | ----------------- | ------- | ------------- | ------ |
| 11513410      | 1147087669482550   | L                  | CC   | 2020              | 2016            | 4                  | 4                 | 3      | 1             | Caucasian or White  |
| 11513410      | 1147087669482550   | L                  | MLO   | 2020              | 2016            | 4                  | 4                 | 3       | 1             | Caucasian or White  |
| 11513410      | 1147087669482550   | R                  | CC   | 2020              | 2016            | 4                  | 4                 | 3      | 1             | Caucasian or White  |
| 11513410      | 1147087669482550   | R                  | MLO   | 2020              | 2016            | 4                  | 4                 | 3       | 1             | Caucasian or White  |
| 11513410      | 1707099151654590   | L                  | CC   | 2020              | 2017            | 3                  | 3                 | 3       | 1             | Caucasian or White  |
| 11513410      | 1707099151654590   | L                  | MLO   | 2020              | 2017            | 3                  | 3                 | 3       | 1             | Caucasian or White  |
| 11513410      | 1707099151654590   | R                  | CC   | 2020              | 2017            | 3                  | 3                 | 3       | 1             | Caucasian or White  |
| 11513410      | 1707099151654590   | R                  | MLO   | 2020              | 2017            | 3                  | 3                 | 3       | 1             | Caucasian or White  |

</div>
</details>



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