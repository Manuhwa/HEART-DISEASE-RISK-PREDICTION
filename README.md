# Minimal HDRPS Streamlit App

This is a minimal **Heart Disease Risk Prediction System (HDRPS)** built with Streamlit. It estimates the likelihood of coronary heart disease from 13 routine clinical and test variables—the same features used to train the saved Logistic Regression model.

## What it does

- Collects **13 patient inputs** (demographics, symptoms, vitals, labs, ECG, and imaging-related codes).
- Loads saved artifacts from `model_artifacts/model.pkl` and `model_artifacts/scaler.pkl`.
- Returns:
  - **Predicted class** — `1` = heart disease likely present, `0` = likely absent
  - **Probability** — modeled chance of heart disease (0–100%)
  - **Risk band** — Low / Moderate / High (from probability thresholds)
  - **Factor analysis** — which inputs pushed the score up or down
- Provides an in-app **About** page with a short variable table; this README is the full field reference.

## Project files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app (Predict + About pages) |
| `model_artifacts/model.pkl` | Trained Logistic Regression model |
| `model_artifacts/scaler.pkl` | Fitted `StandardScaler` (same feature order as training) |
| `model_artifacts/metadata.json` | Feature order, dataset size, cross-validation and test metrics |
| `save_model_artifacts.ipynb` | Notebook to regenerate artifacts from `Dataset/heart.csv` |
| `Dataset/heart.csv` | Training data (required to retrain; not needed only to run inference) |

## Setup

1. Use your main Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). Use **Predict** to enter patient data and **About** for a compact variable list.

---

## Abbreviations and terms

Units, clinical shorthand, and dataset column names used in this README and in the app.

### Units of measurement

| Abbreviation | Full name | Meaning |
|--------------|-----------|---------|
| **mg/dL** | milligrams per deciliter | Concentration in blood (e.g. cholesterol, glucose). **1 dL = 100 mL.** Example: cholesterol `212` means 212 mg of cholesterol per deciliter of blood. |
| **mmHg** | millimeters of mercury | Unit of **pressure**, used for blood pressure. Named after the height of a mercury column in a sphygmomanometer; still standard in clinical BP readings (e.g. `125` mmHg resting systolic). |
| **mm** | millimeters | Here, **millimeters of ST deviation** on an ECG (`oldpeak`)—how far the ST segment is depressed relative to baseline, not a physical length on the body. |
| **bpm** | beats per minute | Heart rate (e.g. maximum heart rate achieved during stress testing, `thalach`). |
| **%** | percent | Probability of heart disease on the dashboard (0–100%). |

### Clinical and test abbreviations

| Abbreviation | Full name / expansion | Meaning in this project |
|--------------|----------------------|-------------------------|
| **HDRPS** | Heart Disease Risk Prediction System | This Streamlit application. |
| **BP** | blood pressure | Usually **resting** pressure in `trestbps` (mmHg). |
| **ECG** / **EKG** | electrocardiogram | Recording of the heart’s electrical activity. **Resting ECG** = `restecg`; exercise-related ST measures = `oldpeak`, `slope`. |
| **ST** | ST segment | Part of the ECG waveform between the S wave and T wave; **ST depression** (`oldpeak`) and **ST slope** (`slope`) describe exercise stress findings. |
| **ST-T** | ST segment and T wave | Combined repolarization pattern; **ST-T abnormality** = `restecg` code `1`. |
| **LV / LVH** | left ventricle / left ventricular hypertrophy | Thickening of the heart’s main pumping chamber; **LV hypertrophy** on resting ECG = `restecg` code `2`. |
| **Angina** | — (Latin: tight chest) | Chest discomfort from **reduced blood flow to the heart**, often with exertion. **Exercise angina** = `exang`. |
| **FBS** | fasting blood sugar | Glucose after fasting; **`fbs`** is coded yes/no for **> 120 mg/dL**, not the exact glucose value. |
| **Fluoroscopy** | — | X-ray imaging during catheterization; **`ca`** counts major coronary vessels seen with contrast. |
| **PET / SPECT** | positron emission tomography / single-photon emission computed tomography | Nuclear imaging used to assess perfusion; reports may inform **`thal`** coding. |

### Dataset column names (model inputs)

| Code | Column | Stands for (approximate) |
|------|--------|---------------------------|
| **cp** | chest pain type | Type of chest pain (0–3). |
| **trestbps** | resting blood pressure | Resting BP (mmHg). |
| **chol** | cholesterol | Serum cholesterol (mg/dL). |
| **fbs** | fasting blood sugar | Fasting glucose > 120 mg/dL (0/1). |
| **restecg** | resting ECG | Resting electrocardiogram result (0–2). |
| **thalach** | maximum heart rate achieved | Peak heart rate on stress test (bpm). |
| **exang** | exercise angina | Angina during exercise (0/1). |
| **oldpeak** | ST depression | Exercise-induced ST depression (mm). |
| **slope** | ST slope | Slope of peak exercise ST segment (1–3). |
| **ca** | coronary arteries | Number of major vessels involved (0–3). |
| **thal** | thalassemia / perfusion | Perfusion defect category (3, 6, 7). |

### Model and statistics (optional)

| Abbreviation | Meaning |
|--------------|---------|
| **ROC-AUC** | Area under the receiver operating characteristic curve; measures how well the model ranks patients with vs without disease (higher is better; max 1.0). |
| **CV** | cross-validation | Repeated train/validate splits to estimate generalization. |

---

## Patient input fields (complete reference)

The model expects **exactly these 13 variables**, in this order. Names in parentheses are the **column names** in `heart.csv` and in the saved model. Most categorical fields use **numeric codes**, not free text—enter the code shown in the app dropdown.

**Important:** This app is **decision support only**. Values must come from real clinical assessment, labs, ECG, and imaging—not guessed. Wrong codes (especially for ECG, slope, vessels, and thalassemia) will produce unreliable predictions.

### Quick reference table

| # | Field name (column) | UI label | Type | Allowed range / codes |
|---|---------------------|----------|------|------------------------|
| 1 | `age` | Age | Number | 20–100 (years) |
| 2 | `sex` | Sex | Category | `0` = Female, `1` = Male |
| 3 | `cp` | Chest Pain Type | Category | `0`–`3` (see below) |
| 4 | `trestbps` | Resting BP | Number | 80–220 (mmHg) |
| 5 | `chol` | Cholesterol | Number | 100–600 (mg/dL) |
| 6 | `fbs` | Fasting Blood Sugar | Category | `0` = ≤120 mg/dL, `1` = >120 mg/dL |
| 7 | `restecg` | Resting ECG | Category | `0`–`2` (see below) |
| 8 | `thalach` | Max Heart Rate | Number | 60–220 (bpm) |
| 9 | `exang` | Exercise Angina | Category | `0` = No, `1` = Yes |
| 10 | `oldpeak` | ST Depression | Number | 0.0–6.0 (mm) |
| 11 | `slope` | ST Slope | Category | `1`, `2`, `3` (see below) |
| 12 | `ca` | Major Vessels | Category | `0`–`3` (count) |
| 13 | `thal` | Thalassemia | Category | `3`, `6`, `7` (see below) |

---

### 1. Age (`age`)

**What it is:** Patient age in **completed years** at the time of assessment.

**Why it matters:** Cardiovascular risk generally increases with age; age is a standard factor in heart-disease risk models.

**How to enter:** Whole number (e.g. `52`).

**Typical range in app:** 20–100.

---

### 2. Sex (`sex`)

**What it is:** Biological sex as recorded in the dataset (binary encoding used for modeling).

**Codes:**

| Code | Meaning |
|------|---------|
| `0` | Female |
| `1` | Male |

**Note:** The underlying Cleveland-style dataset uses this simple encoding; the model does not capture additional sex/gender categories.

---

### 3. Chest pain type (`cp`)

**What it is:** The **type of chest pain** (or lack of pain) that best describes the patient’s presentation. This is a clinical classification, not a diagnosis by itself.

**Codes:**

| Code | Label | Meaning |
|------|-------|---------|
| `0` | Typical angina | Chest discomfort consistent with **classic angina**—often pressure/squeezing, related to exertion, may improve with rest. |
| `1` | Atypical angina | Chest symptoms **somewhat like angina** but with features that are not fully classic (e.g. sharp pain, variable relation to exertion). |
| `2` | Non-anginal pain | Chest pain **unlikely to be angina** (e.g. pleuritic, musculoskeletal, or clearly non-cardiac pattern). |
| `3` | Asymptomatic | **No chest pain** reported; patient may still have disease detected by other tests. |

**Why it matters:** Typical angina and some asymptomatic presentations are associated with higher likelihood of significant coronary disease in historical datasets.

---

### 4. Resting blood pressure (`trestbps`)

**What it is:** **Resting systolic blood pressure** (or the dataset’s resting BP field) measured in **mmHg** at rest, before exercise testing.

**Why it matters:** Hypertension is a major cardiovascular risk factor and affects both disease presence and management targets.

**How to enter:** Integer mmHg (e.g. `125`, `140`).

**Typical range in app:** 80–220. Values **>180** trigger a soft warning in the app.

**Clinical note:** Long-term targets for many adults are **<130/80 mmHg**; this field is only the resting value used by the model, not a treatment plan.

---

### 5. Serum cholesterol (`chol`)

**What it is:** **Serum cholesterol** in **mg/dL** (often total cholesterol or the lab value stored in the record—use the same unit as in the patient chart).

**Why it matters:** Elevated cholesterol contributes to atherosclerosis and coronary artery disease risk.

**How to enter:** Integer mg/dL (e.g. `212`).

**Typical range in app:** 100–600. Values **>300** trigger a soft warning.

**Clinical note:** Many guidelines treat **≥200 mg/dL** as elevated for risk discussion; the app’s recommendations may mention levels ≥200.

---

### 6. Fasting blood sugar (`fbs`)

**What it is:** Whether **fasting blood sugar** is **greater than 120 mg/dL** (a simple diabetes-related flag in this dataset).

**Codes:**

| Code | Meaning |
|------|---------|
| `0` | No — fasting blood sugar **≤ 120 mg/dL** |
| `1` | Yes — fasting blood sugar **> 120 mg/dL** |

**Why it matters:** Diabetes and impaired glucose tolerance increase cardiovascular risk.

**How to obtain:** From fasting glucose lab on the day of (or before) assessment, per your lab’s definition of “fasting.”

---

### 7. Resting electrocardiogram (`restecg`)

**What it is:** Summary of the **resting 12-lead ECG** before exercise, reduced to three categories as in the UCI Cleveland heart disease database.

**Codes:**

| Code | Label | Meaning |
|------|-------|---------|
| `0` | Normal | Resting ECG without significant ST-T abnormality or LVH pattern used in this coding scheme. |
| `1` | ST-T wave abnormality | Resting ECG shows **ST segment and/or T wave abnormalities** (e.g. ischemic or nonspecific repolarization changes)—requires cardiologist/physician interpretation. |
| `2` | LV hypertrophy | ECG suggests **left ventricular hypertrophy** (“probable” or “definite” by the criteria used when the record was created). |

**Why it matters:** Resting ECG abnormalities can reflect strain, ischemia, or hypertrophy associated with heart disease.

**How to enter:** Use the code that matches the **official read** on the resting ECG, not a guess from symptoms alone.

---

### 8. Maximum heart rate achieved (`thalach`)

**What it is:** **Maximum heart rate achieved** during exercise testing (or the peak rate recorded in the file), in **beats per minute (bpm)**.

**Often seen on:** Exercise stress test report as peak HR.

**Why it matters:** Lower peak heart rate for a given age/fitness can reflect medication (e.g. beta-blockers), deconditioning, or cardiovascular limitation; the model was trained on this numeric peak value.

**How to enter:** Integer bpm (e.g. `168`).

**Typical range in app:** 60–220. Very low values (**<90**) trigger a soft warning—confirm whether the patient was on rate-limiting drugs or stopped early.

**Rough reference (not used by the model):** Estimated max HR ≈ `220 − age` is a common rule of thumb; actual peak depends on fitness, medications, and test protocol.

---

### 9. Exercise induced angina (`exang`)

**What it is:** Whether the patient developed **angina (chest discomfort) during exercise** testing.

**Codes:**

| Code | Meaning |
|------|---------|
| `0` | No |
| `1` | Yes |

**Why it matters:** Angina provoked by exertion is a classic sign of **myocardial ischemia** and is strongly relevant to coronary disease workup.

**How to obtain:** From the stress test report or clinician observation during the test.

---

### 10. ST depression induced by exercise (`oldpeak`)

**What it is:** **`oldpeak`** — the amount of **ST segment depression** induced by exercise relative to rest, measured on the ECG during/after stress testing. In the Cleveland dataset this is typically expressed in **millimeters (mm)** of ST deviation (lead and measurement protocol follow the original study conventions).

**What ST depression means:** The **ST segment** is the portion of the ECG between the S wave and the T wave. **Depression** of the ST segment during stress can indicate **ischemia** (insufficient blood flow in a coronary artery territory). Larger values generally indicate more pronounced exercise-induced ST changes (interpretation is clinical).

**How to enter:** Decimal number from **0.0** to **6.0** (e.g. `1.0`, `3.1`). Use **`0.0`** if no exercise-induced ST depression was reported.

**Typical range in app:** 0.0–6.0. Values **>3.0** trigger a soft warning.

**Do not confuse with:** `slope` (below)—`oldpeak` is **how much** the ST segment drops; `slope` is **the shape/direction** of the ST segment at peak exercise.

---

### 11. ST slope (`slope`) — slope of peak exercise ST segment

**What it is:** The **slope (shape) of the ST segment at peak exercise** on the stress ECG—not blood pressure “slope” and not cholesterol trend. It describes **how the ST segment deviates** at maximal or peak stress: upward, flat, or downward. This is a standard variable in the Cleveland heart disease database and is entered as a **category code**, not degrees or mm.

**Clinical background:** During exercise stress testing, the ST segment response helps stratify ischemia. An **upsloping** ST segment at peak exercise is often considered less specific for ischemia than a **flat** or **downsloping** pattern, which are more concerning in many protocols (final interpretation belongs to a qualified clinician).

**Codes used in this app** (must match training data):

| Code | Label | Meaning |
|------|-------|---------|
| `1` | Upsloping | ST segment at peak exercise trends **upward** (generally less suggestive of ischemia than flat/downsloping in classic teaching). |
| `2` | Flat | ST segment at peak exercise is **flat** (horizontal)—often considered more abnormal than upsloping. |
| `3` | Downsloping | ST segment at peak exercise trends **downward**—often considered the most concerning pattern among these three. |

**How to obtain:** From the **cardiology stress test report** or ECG interpretation at peak exercise—use the coded category documented for the patient, mapped to `1` / `2` / `3` as above.

**Relation to `oldpeak`:**  
- **`oldpeak`** = magnitude of ST **depression** (numeric, mm).  
- **`slope`** = **morphology** of the ST segment at peak exercise (categorical).  
A patient can have low `oldpeak` but an abnormal `slope`, or vice versa; both were separate inputs when the model was trained.

**Note on raw data:** Some rows in `heart.csv` may show other slope encodings (e.g. `0`) from source merges; the Streamlit app only accepts **`1`, `2`, `3`** to stay consistent with the trained scaler and model.

---

### 12. Number of major vessels (`ca`)

**What it is:** **`ca`** — the number of **major coronary vessels** (0–3) showing significant disease on **fluoroscopy** (contrast visualization of the coronary arteries), as recorded in the Cleveland dataset. It is a **count of vessels**, not a anatomy diagram label in the UI.

**Codes:**

| Code | Meaning |
|------|---------|
| `0` | No major vessels with significant disease by the study’s fluoroscopy criteria |
| `1` | One major vessel involved |
| `2` | Two major vessels involved |
| `3` | Three major vessels involved |

**Why it matters:** Multi-vessel disease is associated with higher burden of coronary atherosclerosis.

**How to obtain:** From cardiac catheterization / angiography or the fluoroscopy field documented in the patient record—not from non-invasive CT alone unless your workflow maps to this same variable.

**Caution:** In real practice this comes from invasive angiography; if unknown, do not invent a value—prediction quality depends on accurate inputs.

---

### 13. Thalassemia / perfusion defect (`thal`)

**What it is:** In the Cleveland database, **`thal`** is labeled “thalassemia” but in practice it encodes **thalassemia or nuclear perfusion defect status** (fixed vs reversible defect vs normal). It reflects **imaging/perfusion** information, not a complete hematology workup for thalassemia alone.

**Codes:**

| Code | Label | Meaning |
|------|-------|---------|
| `3` | Normal | No defect / normal study in this coding scheme |
| `6` | Fixed defect | **Fixed perfusion defect** (e.g. scar from prior infarction)—does not improve with stress/rest protocol used in testing |
| `7` | Reversible defect | **Reversible perfusion defect**—suggests inducible ischemia (blood flow limitation under stress that improves at rest) |

**Why it matters:** Reversible defects point toward ischemia; fixed defects point toward scar or permanent damage—both alter cardiovascular risk assessment.

**How to obtain:** From the nuclear stress test, PET/SPECT report, or the coded field in the patient’s angiography/imaging summary as used in your source data.

---

## Prediction output (how to read results)

| Output | Meaning |
|--------|---------|
| Class `1` | Model predicts **heart disease present** (probability ≥ 50%) |
| Class `0` | Model predicts **heart disease absent** (probability < 50%) |
| Probability | Estimated chance of disease (class `1`); shown as % on the dashboard |
| Confidence | For class `1`, confidence ≈ probability; for class `0`, ≈ `100% − probability` |
| Risk band | Low (&lt;30%), Moderate (30–70%), High (≥70%) — UI bands only, not a clinical staging system |
| Top contributing factors | Logistic regression: which **scaled** inputs pushed toward disease vs away |

Soft warnings (high BP, high cholesterol, low max HR, high ST depression) are **plausibility checks**, not diagnoses.

---

## Model and data notes

- **Algorithm:** Logistic Regression on standardized features (`StandardScaler` fit on training data).
- **Feature order** must match `metadata.json` / `FEATURE_ORDER` in `app.py` (see table above).
- **Retraining:** Run `save_model_artifacts.ipynb` with `Dataset/heart.csv` present.
- **Test performance** (holdout, from `metadata.json`): accuracy ≈ 0.81, ROC-AUC ≈ 0.93 — performance on **your** population may differ.
- **Disclaimer:** HDRPS does **not** replace clinical judgment, physical exam, or full diagnostic workup. Use only for education or decision support where appropriate and permitted by policy.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `Model artifacts not found` | Run the notebook to create `model_artifacts/model.pkl` and `scaler.pkl` |
| Dropdown won’t accept my slope value | App allows only `1`, `2`, `3`; map from the stress ECG report |
| Results seem wrong | Verify coded fields (`cp`, `restecg`, `slope`, `ca`, `thal`) against source documents, not symptoms alone |
| Retrain fails | Ensure `Dataset/heart.csv` exists and has all 13 feature columns plus `target` |
