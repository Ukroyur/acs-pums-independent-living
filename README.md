
# Independent Living Difficulty Classification — ACS 2023 PUMS (36‑Model Study)

**Course:** BU MET CS699 — Data Mining *(Spring 2025)*  
**Instructor:** Dr. Jae Young Lee  
**Authors:** Utkarsh Roy (BU ID: U69501306), Siddhraj Parmar (BU ID: U35065911)  
**Status:** Final Report & Reproducible Code  
**Last Updated:** 2025-10-02

---

## 🔎 Project Overview
This project predicts **difficulty in independent living** using the **2023 American Community Survey (ACS) PUMS** 1‑Year dataset. We built a **full machine‑learning pipeline** with rigorous **preprocessing → class balancing → feature selection → model training → hyperparameter tuning → evaluation**.

We evaluated **36 models** across:
- **6 classifiers**: Logistic Regression (GLM), Elastic Net (GLMNet), SVM (RBF), Neural Network (nnet), Naive Bayes, K‑Nearest Neighbors (KNN)  
- **3 feature selection methods**: **Boruta**, **Information Gain**, **LASSO**  
- **2 class balancing strategies**: **Undersampling** and **SMOTE** (training set only)

**Headline result:** The best model was **SVM (RBF)** trained on **SMOTE‑balanced** data with **Boruta** features, yielding **TPR≈0.93** for *Class=No* and **TPR≈0.90** for *Class=Yes* on the **test set**, with strong overall balance (ROC ≈ **0.80**). A close second was **SVM + LASSO + SMOTE** (Yes‑class TPR ≈ **0.875**).

---

## 📦 Data
- **Source:** 2023 ACS PUMS, 1‑Year release  
- **Download:** <https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/>  
- **Raw file used:** `project_data.csv` *(modified ACS slice)*  
- **Observations:** ~**4,318**  
- **Variables:** **117**  
- **Target:** `Class` — **1 = Yes** (difficulty), **0 = No**

> ⚠️ **Data policy:** Do **not** commit large/raw ACS data. Keep a **small sample** for tests. Document full download steps in `DATA.md`.

---

## 🗂️ Repository Layout
```
acs-pums-independent-living/
├─ README.md
├─ DATA.md                     # where to download data + expected schema
├─ LICENSE                     # MIT recommended
├─ .gitignore                  # R, Rproj, CSVs, cache, figures, etc.
├─ data/
│  ├─ raw/                     # (ignored) original ACS files
│  ├─ sample/                  # small sample for demo
│  └─ processed/               # project_data_cleaned.csv, splits
├─ src/
│  ├─ prep/                    # loading, missing values, nzv, scaling, correlation
│  ├─ balance/                 # undersample.R, smote.R
│  ├─ features/                # boruta.R, info_gain.R, lasso.R
│  ├─ train/                   # glm.R, glmnet.R, svm.R, nn.R, nb.R, knn.R
│  └─ eval/                    # evaluate.R, aggregate.R, utils_metrics.R
├─ results/
│  ├─ summary.csv              # all metrics across 36 models
│  └─ confmats/                # confusion matrices
├─ figures/                    # ROC curves, heatmaps, bars
└─ reports/
   └─ final_report.pdf         # your PDF
```

---

## 🧰 Tools & Libraries (R)
`tidyverse`, `dplyr`, `ggplot2`, `caret`, `rsample`, `corrplot`, `reshape2`, `ROSE`, `smotefamily`,  
`Boruta`, `FSelector`, `glmnet`, `pROC`, `nnet`, `e1071`, `naivebayes`, `gbm`, `kernlab`

Create an `R` script to install:
```r
packages <- c(
  "tidyverse","dplyr","ggplot2","caret","rsample","corrplot","reshape2",
  "ROSE","smotefamily","Boruta","FSelector","glmnet","pROC","nnet",
  "e1071","naivebayes","gbm","kernlab"
)
to_install <- setdiff(packages, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
```

---

## ⚙️ End‑to‑End Reproduction (Quickstart)
```r
# 0) set working directory to repo root
setwd("acs-pums-independent-living")

# 1) Load + Inspect
source("src/prep/load_and_inspect.R")        # reads project_data.csv, str/summary

# 2) Preprocess
source("src/prep/missing_values.R")          # drop cols > 2000 NAs, prop-based impute
source("src/prep/nzv_normalize_corr.R")      # remove nzv, center/scale numerics, drop corr > 0.8
source("src/prep/save_clean.R")              # writes data/processed/project_data_cleaned.csv

# 3) Split
source("src/prep/split.R")                   # 65/35 stratified split → initial_train.csv / initial_test.csv

# 4) Balance (choose one or run both for experiments)
source("src/balance/undersample.R")          # ROSE::ovun.sample → balanced undersampled train
source("src/balance/smote.R")                # smotefamily::SMOTE → balanced SMOTE train

# 5) Feature Selection (run on each balanced train set)
source("src/features/boruta.R")              # → selected_boruta.txt
source("src/features/info_gain.R")           # → selected_ig.txt
source("src/features/lasso.R")               # → selected_lasso.txt

# 6) Train + Tune (5‑fold CV; ROC as metric)
source("src/train/glm.R")
source("src/train/glmnet.R")
source("src/train/svm.R")
source("src/train/nn.R")
source("src/train/nb.R")
source("src/train/knn.R")

# 7) Evaluate
source("src/eval/evaluate.R")                # confusionMatrix, AUC, MCC, Kappa, TPR/FPR by class
source("src/eval/aggregate.R")               # builds results/summary.csv (all 36 combos)
```

---

## 🧼 Data Preprocessing (Key Steps)
1. **Load** `project_data.csv`; inspect with `str()` and `summary()`  
2. **Drop columns** with **>2000 missing** values  
3. **Impute** remaining NAs using **proportion‑based sampling** to preserve distributions  
   - Special cases: `JWMNP`, `WKWN` → impute **0** by domain logic  
4. **Remove** identifier (`SERIALNO`) and **near‑zero variance** features (`caret::nearZeroVar`)  
5. **Normalize** numerics (`caret::preProcess(method=c("center","scale"))`)  
6. **Correlation filter:** remove variables with |r| **> 0.8** (`findCorrelation`)  
7. **Save** as `data/processed/project_data_cleaned.csv`  

**Final clean dataset:** ~**38** variables, 0 missing.

---

## 🔀 Train/Test Split
- **Stratified** split on `Class` via `rsample::initial_split`  
- **65% train / 35% test**  
- Outputs: `initial_train.csv`, `initial_test.csv`

---

## ⚖️ Handling Class Imbalance
- **Undersampling** (ROSE::`ovun.sample`) — reduce majority (Class=0)  
- **SMOTE** (smotefamily::`SMOTE`) — synthesize minority (Class=1)  
> Balancing is performed **only on the training set** to avoid leakage; the **test set is kept original**.

---

## 🧭 Feature Selection
- **Boruta** — all‑relevant wrapper around Random Forest; retains weak but useful predictors  
- **Information Gain** — univariate entropy‑based ranking (FSelector)  
- **LASSO** — embedded selection via `glmnet` (non‑zero coefficients retained)  
Each method is applied **separately** to **undersampled** and **SMOTE** datasets, yielding multiple feature subsets.

---

## 🤖 Models & Tuning
All models use **5‑fold CV** with **ROC** as the primary metric. Most include **centering/scaling**.

- **GLM (Logistic Regression)** — baseline (no tuning)  
- **GLMNet (Elastic Net)** — grid over `alpha ∈ [0..1]`, `lambda ∈ 10^-6..10^0`  
- **SVM (RBF)** — grid over `C` and `sigma` (`kernlab::sigest` to seed)  
- **Neural Network (nnet)** — grid over `size` ∈ (1, 3, 5), `decay` ∈ (0.1, 0.5, 1, 2)  
- **Naive Bayes** — grid over `laplace`, `usekernel`, `adjust`  
- **KNN** — `k` ∈ (3, 5, 7, 9, 11)

**Evaluation metrics:** TPR, FPR, Precision, Recall, F1, ROC AUC, **MCC**, **Cohen’s Kappa** (per class + weighted).

---

## 🧪 Results (Test Set Highlights)

**Top performer:** **SVM (RBF) + Boruta + SMOTE**  
- **Class=No:** TPR ≈ **0.93**, FPR ≈ **0.09**  
- **Class=Yes:** TPR ≈ **0.90**  
- **ROC AUC:** ≈ **0.80**  
- Balanced performance across classes; robust to class imbalance.

**Runner‑up:** **SVM (RBF) + LASSO + SMOTE**  
- **Class=No:** TPR ≈ **0.93**  
- **Class=Yes:** TPR ≈ **0.875**  
- Competitive overall metrics; qualifies for extra credit.

> Full comparative tables for all **36 models** are exported to `results/summary.csv` with confusion matrices under `results/confmats/`.

---

## 📈 Plots to Include
- **Correlation heatmap** (pre‑filter)  
- **ROC curves** overlay by model family  
- **Bar chart** of F1 / TPR for each of 36 models  
- **Feature importance** (Boruta) and **coefficients** (LASSO non‑zeros)

---

## 🧪 Reproducible Metrics Helper
Custom evaluator computes per‑class metrics and aggregates:
```r
source("src/eval/utils_metrics.R")  # calculates TPR, FPR, Precision, Recall, F1, MCC, Kappa, ROC
```

---

## ✅ Key Takeaways
1. **SVM (RBF) consistently dominated** across feature sets when trained on **SMOTE‑balanced** data.  
2. **Feature selection matters**: Boruta and LASSO both yielded strong subsets; Info Gain was competitive.  
3. **Handle imbalance only on train** to avoid leakage; **evaluate on original test**.  
4. **Per‑class metrics** (especially **TPR for Class=Yes**) are essential when the positive class is rare.

---

## 🧩 Limitations & Next Steps
- Low precision for *Class=Yes* indicates **many false positives**; consider **cost‑sensitive SVM**, **threshold tuning**, or **calibration**.  
- Explore **stacked ensembles** and **gradient boosting** baselines (e.g., `xgboost`) with careful tuning.  
- Incorporate **domain features** (interactions, transformations) and **explainability** (SHAP).

---

## 🔐 Ethical Use
ACS microdata can include sensitive socio‑economic signals. Use models responsibly; avoid deployment without fairness checks and stakeholder review.

---

## 📄 Citation
If you use this repository, please cite:
> Roy, U., & Parmar, S. (2025). *Independent Living Difficulty Classification — ACS 2023 PUMS*. BU MET, CS699 Data Mining.

