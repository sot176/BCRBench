# CSV File Format

To use the models and dataset classes in this repository, datasets must be provided as CSV files with a predefined structure.

---

## 📌 Required Columns

| Column Name | Description |
|-------------|-------------|
| `patient_id` | Unique identifier for each patient |
| `exam_id` | Unique identifier for each exam (one per timepoint) |
| `ImageLaterality` | `L` (left) or `R` (right breast) |
| `view` | Mammography view (e.g., CC, MLO) |
| `diagnosed_date_year` | Year of diagnosis (empty if cancer-free) |
| `study_date_year` | Year the mammogram was acquired |
| `Time_to_Cancer_Years` | Years until diagnosis (empty if cancer-free) |
| `years_last_followup` | Years until last follow-up |
| `density` | Breast density (dataset-specific format) |
| `cancer_type` | Cancer type (dataset-specific encoding) |
| `race` | Optional demographic information |

---

## 🧬 Dataset Curation and Label Generation

### EMBED

**Positive cases:**
- BI-RADS = 6  (biopsy-proven cancer)
- Pathological severity = 0 or 1  (invasive or non-invasive cancer)

**Negative cases:**
- BI-RADS = 1 or 2   (negative or benign)
- BI-RADS 0 later reclassified as 1 or 2  

**Column references in the original CSV-files:**
- BI-RADS → `asses`  
- Diagnosis date → `procdate_anon`  

---

### Example (EMBED)

| exam_year | BI-RADS | diagnosis_year | time_to_cancer |
|----------|--------|----------------|----------------|
| 2016 | 1 | — | 3 |
| 2017 | 2 | — | 2 |
| 2018 | 1 | — | 1 |
| 2019 | 6 | 2019 | 0 |

---

### CSAW-CC

**Positive cases:**
- `rad_timing = 1` → screen-detected cancer  
- `rad_timing = 2` → interval cancer  

**Negative cases:**
- No recorded `rad_timing`

**Note:**  
Biopsy dates are not available. `rad_timing` is used as a proxy for cancer timing.

---

### Cancer Year Assignment (CSAW-CC)

- Data grouped by patient ID  
- The latest exam year per patient is selected  
- Screen-detected → same year  
- Interval cancer → exam year + 1  

**Special case:**
- If latest exam year = 2016 → cancer year = 2016  

---

### Example (CSAW-CC)

| exam_year | rad_timing | cancer_year | time_to_cancer |
|----------|------------|-------------|----------------|
| 2013 | — | — | 4 |
| 2014 | — | — | 3 |
| 2015 | — | — | 2 |
| 2016 | 2 | 2017 | 1 |

---

## ⏱ Time-to-Cancer Definition

Time-to-Cancer represents the interval between a screening exam and the corresponding breast cancer diagnosis.

---

## 🧾 Cancer Type Encoding
The column names listed below correspond to the original dataset CSV files.

| Dataset | Column | Value | Meaning |
|--------|--------|-------|--------|
| EMBED | `path_severity` | 0 | Invasive cancer |
| EMBED | `path_severity` | 1 | Non-invasive cancer |
| EMBED | `path_severity` | 2 | High-risk lesion |
| EMBED | `path_severity` | 3 | Borderline lesion |
| EMBED | `path_severity` | 4 | Benign findings |
| EMBED | `path_severity` | 5 | Negative |
| EMBED | `path_severity` | 6 | Non-breast cancer |
| CSAW-CC | `x_type` | 1 | In situ (non-invasive) |
| CSAW-CC | `x_type` | 2 | Invasive ≤ 15 mm |
| CSAW-CC | `x_type` | 3 | Invasive > 15 mm |

---

## 🧾 Breast Density
The column names listed below correspond to the original dataset CSV files.

| Dataset | Column | Value | Meaning |
|--------|--------|-------|--------|
| EMBED | `tissueden` | 1 | Almost entirely fat (BI-RADS A) |
| EMBED | `tissueden` | 2 | Scattered density (BI-RADS B) |
| EMBED | `tissueden` | 3 | Heterogeneously dense (BI-RADS C) |
| EMBED | `tissueden` | 4 | Extremely dense (BI-RADS D) |
| EMBED | `tissueden` | 5 | Normal male |
| CSAW-CC | `libra_percentdensity` | 0–100 | Continuous percentage |

---

## 📊 Examples

### Cancer-Free Patient

<details>
<summary>Show example</summary>

| patient_id | exam_id | laterality | view | diagnosis_year | study_year | time_to_cancer | followup | density | cancer_type | race |
|-----------|--------|------------|------|----------------|------------|----------------|----------|--------|-------------|------|
| 10093833 | 64887706 | L | CC | — | 2015 | — | 3 | 1 | — | White |
| 10093833 | 64887706 | L | MLO | — | 2015 | — | 3 | 1 | — | White |
| 10093833 | 64887706 | R | CC | — | 2015 | — | 3 | 1 | — | White |
| 10093833 | 64887706 | R | MLO | — | 2015 | — | 3 | 1 | — | White |

</details>

---

### Cancer Patient

<details>
<summary>Show example</summary>

| patient_id | exam_id | laterality | view | diagnosis_year | study_year | time_to_cancer | followup | density | cancer_type | race |
|-----------|--------|------------|------|----------------|------------|----------------|----------|--------|-------------|------|
| 11513410 | 11470876 | L | CC | 2020 | 2016 | 4 | 4 | 3 | 1 | White |
| 11513410 | 11470876 | L | MLO | 2020 | 2016 | 4 | 4 | 3 | 1 | White |
| 11513410 | 11470876 | R | CC | 2020 | 2016 | 4 | 4 | 3 | 1 | White |
| 11513410 | 11470876 | R | MLO | 2020 | 2016 | 4 | 4 | 3 | 1 | White |

</details>
