# Breast Cancer Risk Prediction Models

Unified repository implementing state-of-the-art image-based breast cancer risk prediction models. This project aims to simplify training and evaluation of multiple models using a single, unified pipeline, supporting both single- and multi-timepoint mammogram analysis. 

We aim to encourage further research by providing a single, unified repository where all image-based breast cancer risk prediction models can be collected, compared, and extended. By centralizing these implementations, we hope to accelerate progress in the field and make it easier for researchers to benchmark and build upon existing methods.

---

## Table of Contents
1. 📘 [Overview](#overview)  
2. ⚙️ [Models Implemented](#models-implemented)  
3. 📚 [Comparison of Models](#comparison-of-models)  
4. 🗂️ [Datasets](#datasets)
5. 📄 [CSV File Format](#csv-file-format)  
6. ▶️ [Usage](#usage)  
7. 📄 [Citation](#citation) 
8. 📝 [License](#license) 

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

## Comparison of Models
 
- Single-view vs Multi-view: OA-BReaCR, ImgFeatAlign (single-view); LMV-Net (multi-view)

- Single-timepoint vs Multi-timepoint: Mirai (single); VMRA-MAR, OA-BReaCR, ImgFeatAlign, LMV-Net (multi)

- Patient-wise vs Breast-wise aggregation: Mirai, VMRA-MAR (patient-wise, all 4 images of a patient); LMV-Net (breast-wise, per breast)

---

## Datasets
This repository integrates two large, publicly available mammography datasets:
- **Emory Breast Imaging Dataset (EMBED)**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview}
- **Cohort of Screen-Aged Women Case Control (CSAW-CC)**: https://snd.se/en/catalogue/dataset/2021-204-1


---

## CSV File Format
To use the models and dataset classes in this repository, the datasets must be organized as CSV files with specific columns.



<details>
<summary><b>Required Columns</b></summary>
<div style="font-size: 9px">

| Column Name | Description |
|-------------|-------------|
| <sub>`patient_id`</sub> | <sub>Unique ID for each patient</sub> |
| <sub>`exam_id`</sub> | <sub>Unique ID for each exam (one per timepoint)</sub> |
| <sub>`ImageLaterality`</sub> | <sub>`L` for left breast, `R` for right breast</sub> |
| <sub>`view`</sub> | <sub>Mammogram view (e.g., CC, MLO)</sub> |
| <sub>`diagnosed_date_year`</sub> | <sub>Year of breast cancer diagnosis (empty if cancer-free)</sub> |
| <sub>`study_date_year`</sub> | <sub>Year the mammogram was taken</sub> |
| <sub>`Time_to_Cancer_Years`</sub> | <sub>Years until cancer diagnosis (empty if cancer-free)</sub> |
| <sub>`years_last_followup`</sub> | <sub>Years from this exam until the last follow-up for the patient</sub> |
| <sub>`density`</sub> | <sub>Breast density. In EMBED, categorical (1–5). In CSAW-CC, continuous 0–100, which can be grouped into categories.</sub> |
| <sub>`cancer_type`</sub> | <sub>Cancer type. In EMBED, use `path_severity` (0–6). In CSAW-CC, use `x_type` (1–3).</sub> |
| <sub>`race`</sub> | <sub>Patient race (optional, leave empty if not available)</sub> |
</div>
</details>


<details>
<summary><b>Values for Cancer types</b></summary>

The column names listed below correspond to the original dataset CSV files.

| Dataset | Column | Values | Meaning |
| ------- | ------ | ------ | ------- |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>0</sub> | <sub>Invasive cancer</sub> |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>1</sub> | <sub>Non-invasive cancer</sub> |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>2</sub> | <sub>High-risk lesion</sub> |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>3</sub> | <sub>Borderline lesion</sub> |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>4</sub> | <sub>Benign findings</sub> |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>5</sub> | <sub>Negative (normal breast tissue)</sub> |
| <sub>EMBED</sub>   | <sub>path_severity</sub> | <sub>6</sub> | <sub>Non-breast cancer</sub> |
| <sub>CSAW-CC</sub> | <sub>x_type</sub>        | <sub>1</sub> | <sub>In situ only (non-invasive)</sub> |
| <sub>CSAW-CC</sub> | <sub>x_type</sub>        | <sub>2</sub> | <sub>Invasive ≤ 15 mm</sub> |
| <sub>CSAW-CC</sub> | <sub>x_type</sub>        | <sub>3</sub> | <sub>Invasive > 15 mm</sub> |
</details>


<details>
<summary><b>Values for Breast density</b></summary>

The column names listed below correspond to the original dataset CSV files.

| Dataset | Column | Values / Range | Meaning |
| ------- | ------ | -------------- | ------- |
| <sub>EMBED</sub>   | <sub>tissueden</sub> | <sub>1</sub> | <sub>Almost entirely fat (BIRADS A)</sub> |
| <sub>EMBED</sub>   | <sub>tissueden</sub> | <sub>2</sub> | <sub>Scattered fibroglandular densities (BIRADS B)</sub> |
| <sub>EMBED</sub>   | <sub>tissueden</sub> | <sub>3</sub> | <sub>Heterogeneously dense (BIRADS C)</sub> |
| <sub>EMBED</sub>   | <sub>tissueden</sub> | <sub>4</sub> | <sub>Extremely dense (BIRADS D)</sub> |
| <sub>EMBED</sub>   | <sub>tissueden</sub> | <sub>5</sub> | <sub>Normal male</sub> |
| <sub>CSAW-CC</sub> | <sub>libra_percentdensity</sub> | <sub>0–100</sub> | <sub>Percentage breast density. Can be binned into categories (e.g., low, medium, high).</sub> |
</details>


<details>
<summary><b>Example: Cancer-Free Patient</b></summary>

|<sub>patient_id<sub>|<sub>exam_id<sub>|<sub>ImageLaterality<sub>|<sub>view<sub>|<sub>diagnosed_date_year<sub> |<sub>study_date_year<sub> |<sub>Time_to_Cancer_Years<sub>|<sub>years_last_followup<sub>|<sub>density<sub> |<sub>cancer_type<sub>|<sub>race<sub> |
| :--------: | :-----: | :-----------: | :--: | :---------------: | :------------: | :----------------: | :---------------: | :-----: | :----------: | :--: |
| <sub>10093833<sub> | <sub>64887706<sub> | <sub>L<sub> | <sub>CC<sub> |            | <sub>2015<sub>  |                    | <sub>3<sub>  |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
| <sub>10093833<sub> | <sub>64887706<sub> |<sub>L<sub>  | <sub>MLO<sub> |                   | <sub>2015<sub>|                    |<sub>3<sub>  |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
| <sub>10093833<sub> | <sub>64887706<sub> | <sub>R<sub>   |<sub>CC<sub>|                   | <sub>2015<sub>|                    |<sub>3<sub>  |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
| <sub>10093833<sub> | <sub>64887706<sub>| <sub>R<sub>  |<sub>MLO<sub> |                   | <sub>2015<sub>|                    |<sub>3<sub>  |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
| <sub>10093833<sub> | <sub>41737690<sub> |<sub>L<sub>  |<sub>CC<sub>|                   | <sub>2015<sub>|                    | <sub>0<sub> |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
| <sub>10093833<sub> |<sub>41737690<sub> |<sub>L<sub> |<sub>MLO<sub> |                   | <sub>2015<sub>|                    |<sub>0<sub>|<sub>1<sub>|             |<sub>Caucasian or White<sub>  |
| <sub>10093833<sub> | <sub>41737690<sub> | <sub>R<sub>  |<sub>CC<sub>|                   | <sub>2015<sub> |                    |<sub>0<sub>|<sub>1<sub>|             |<sub>Caucasian or White<sub>  |
| <sub>10093833<sub> | <sub>41737690<sub> |  <sub>R<sub>  |<sub>MLO<sub> |                   | <sub>2015<sub> |                    | <sub>0<sub> |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
</details>


<details>
<summary><b>Example: Cancer Patient</b></summary>

|<sub>patient_id<sub>|<sub>exam_id<sub>|<sub>ImageLaterality<sub>|<sub>view<sub>|<sub>diagnosed_date_year<sub> |<sub>study_date_year<sub> |<sub>Time_to_Cancer_Years<sub>|<sub>years_last_followup<sub>|<sub>density<sub> |<sub>cancer_type<sub>|<sub>race<sub> |
| :--------: | :-----: | :-----------: | :--: | :---------------: | :------------: | :----------------: | :---------------: | :-----: | :----------: | :--: |
| <sub>11513410<sub> | <sub>11470876<sub> | <sub>L<sub> | <sub>CC<sub> | <sub>2020<sub> | <sub>2016<sub>| <sub>4<sub>| <sub>4<sub> | <sub>3<sub>| <sub>1<sub>| <sub>Caucasian or White<sub>  |
| <sub>11513410<sub>|<sub>11470876<sub> |<sub>L<sub> | <sub>MLO<sub>  | <sub>2020<sub> | <sub>2016<sub>| <sub>4 <sub> | <sub>4 <sub> | <sub>3<sub> |<sub>1<sub>| <sub>Caucasian or White<sub>|
| <sub>11513410<sub>|<sub>11470876<sub> | <sub>R<sub> | <sub>CC<sub> | <sub>2020<sub> | <sub>2016<sub>| <sub>4 <sub> | <sub>4 <sub> | <sub>3<sub> |<sub>1<sub>| <sub>Caucasian or White<sub>|
| <sub>11513410<sub>| <sub>11470876<sub> |<sub>R<sub>| <sub>MLO<sub> | <sub>2020<sub> | <sub>2016<sub>| <sub>4 <sub> |  <sub>4 <sub> | <sub>3<sub> | <sub>1<sub>| <sub>Caucasian or White<sub>|
|<sub>11513410<sub>| <sub>17070991<sub>     |<sub>L<sub> |<sub>CC<sub> | <sub>2020<sub> | <sub>2017<sub>  | <sub>3<sub>| <sub>3<sub> | <sub>3<sub> |<sub>1<sub>| <sub>Caucasian or White<sub>|
|<sub>11513410<sub>| <sub>17070991<sub>|<sub>L<sub> |<sub>MLO<sub>| <sub>2020<sub> | <sub>2017<sub> | <sub>3<sub>| <sub>3<sub> | <sub>3<sub> |<sub>1<sub>| <sub>Caucasian or White<sub>|
|<sub>11513410<sub> | <sub>17070991<sub>|<sub>R<sub>|<sub>CC<sub> | <sub>2020<sub> | <sub>2017<sub> | <sub>3<sub>| <sub>3<sub> | <sub>3<sub> |<sub>1<sub>| <sub>Caucasian or White<sub> |
| <sub>11513410<sub>| <sub>17070991<sub>| <sub>R<sub> | <sub>MLO<sub>| <sub>2020<sub>  | <sub>2017<sub> | <sub>3<sub> | <sub>3<sub> | <sub>3<sub> | <sub>1<sub>| <sub>Caucasian or White<sub>|

</details>


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
@misc{breast_cancer_risk_github,
  title = {BreastCancerRiskBenchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sot176/BreastCancerRiskBenchmark}}
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