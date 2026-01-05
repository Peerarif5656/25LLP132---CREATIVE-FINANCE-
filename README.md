# Predicting Bank Customer Churn Using Machine Learning (25LLP132)

**Module:** 25LLP132 – Principles of Artificial Intelligence and Data Analytics  
**Group:** Creative Finance  
**Dataset:** Kaggle “Churn Modelling” (`Churn_Modelling.csv`)  
**Task:** Predict customer churn (`Exited`) to support early retention intervention.

## Project Goal (Business Context)
Customer churn is costly: retaining customers is usually cheaper than acquiring new ones.  
This project builds a supervised ML workflow to identify high-risk customers early so retention teams can take targeted actions.

## Dataset Summary
- Rows: 10,000 customers  
- Target: `Exited` (1 = churned, 0 = retained)  
- Removed non-predictive identifiers: `RowNumber`, `CustomerId`, `Surname`  
- Features include demographic, financial, and behavioural variables  
- Churn rate is ~20.37%, so evaluation uses imbalance-aware metrics (Recall, F1, PR-AUC)

## Reproducible Pipeline (Leakage-Safe)
We implement a leakage-safe machine learning pipeline:

**Preprocess → SMOTE → Model**

- Preprocess: `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- SMOTE: applied **inside the pipeline**, so oversampling happens only on training folds (no leakage)
- Split: 70% train / 30% test (stratified)
- Validation: split from training only (threshold exploration); test set untouched until final evaluation
- Models compared:
  - Logistic Regression (baseline)
  - Decision Tree (interpretable)
  - Random Forest (300 trees)
  - Gradient Boosting (best overall in our run)
- Stability check: 5-fold stratified cross-validation

## Repository Contents
- `churn_pipeline.py` — end-to-end pipeline to reproduce figures and tables
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes dataset and generated outputs from GitHub
- `report_figures/` — generated locally when you run the script (not committed)

## How to Run
### 1) Put the dataset in the same folder
Download the Kaggle Churn Modelling dataset and place:
- `Churn_Modelling.csv`

### 2) Install dependencies
```bash
pip install -r requirements.txt

bash
python churn_pipeline.py


