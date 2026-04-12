
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
<summary><b>Dataset Curation and Label Generation</b></summary>

### EMBED labelling
**Positive cases:**
- BI-RADS score = 6 (biopsy-proven cancer)
- Pathological severity score = 0 or 1 (invasive or non-invasive cancer)


**Negative cases:**
- BI-RADS scores = 1 or 2 (negative or benign)
- BI-RADS 0 cases later reclassified as BI-RADS 1 or 2


**Column references:**
- BI-RADS scores correspond to the column `asses` in the original CSV file

**Cancer diagnosis date:**
- Defined as the pathology result date (biopsy confirmation)
- Corresponds to the column `procdate_anon` in the original CSV file

#### 🧾 Example (EMBED – One Patient)
| exam_year | asses (BI-RADS) | procdate_anon    | time_to_cancer |
|----------|----------------|---------------|----------------|
| 2016     | 1              | —               | 3 |
| 2017     | 2              | —               | 2 |
| 2018     | 1              | —               | 1 |
| 2019     | 6              | 2019-06-15      | 0 |

- Cancer is confirmed in 2019 (`procdate_anon`)  
- Time-to-Cancer is computed backward for prior screenings:
  - 2016 → 3 years  
  - 2017 → 2 years  
  - 2018 → 1 year  
- The cancer-year exam (2019) has Time-to-Cancer = 0


### CSAW-CC labeling

**Positive cases (based on `rad_timing`):**
- `rad_timing` = 1 → Screen-detected cancer
- `rad_timing` = 2 → Interval cancer

**Negative cases:**
- No recorded `rad_timing`
- Indicates no cancer diagnosis during the study period

**Important note:**
Biopsy dates are not available
`rad_timing` is used as a proxy for cancer timing


**Cancer Year Assignment (CSAW-CC)**
- Data grouped by patient ID
- The latest exam year per patient is selected
- Screen-detected (rad_timing = 1) → Same year as exam
- Interval cancer (rad_timing = 2) → Exam year + 1

Special case:
- If latest exam year = 2016 → Cancer year set to 2016
(Final year of CSAW-CC study)


#### 🧾 Example (CSAW-CC – One Patient)
| exam_year | rad_timing | cancer_year      | time_to_cancer |
|----------|------------|-------------|----------------|
| 2013     | —          | —        | 4 |
| 2014     | —          | —         | 3 |
| 2015     | —          | —           | 2 |
| 2016     | 2          | 2017       | 1 |

- Latest exam: 2016 with `rad_timing = 2` (interval cancer)  
- Cancer year = 2016 + 1 = 2017  
- Time-to-Cancer is computed as:
  - 2013 → 4 years  
  - 2014 → 3 years  
  - 2015 → 2 years  
  - 2016 → 1 year  

### Time-to-Cancer Definition
- The Time-to-Cancer label represents the time interval between:

- The date of a screening mammogram and the corresponding breast cancer diagnosis.
<sub>2015<sub> |                    | <sub>0<sub> |<sub>1<sub>|             | <sub>Caucasian or White<sub>  |
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