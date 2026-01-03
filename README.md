# Predicting Bank Customer Churn Using Machine Learning

**Group:** Creative Finanace  
**Module:** 25LLP132 – Principles of Artificial Intelligence and Data Analytics  
**Project:** Predicting Bank Customer Churn Using Machine Learning  
**Dataset:** Kaggle “Churn Modelling” (`Churn_Modelling.csv`)  

This repository contains the reproducible pipeline used in our **executive report** and **video presentation**.

---

## Project Goal (Business Context)

Customer churn is costly: retaining customers is usually cheaper than acquiring new ones.  
Our goal is to **identify high-risk customers early** so the bank can apply **targeted retention strategies** rather than contacting all customers.

---

## Dataset Summary

- **Rows:** 10,000 customers  
- **Target:** `Exited` (1 = churned, 0 = retained)  
- Removed non-predictive identifiers: `RowNumber`, `CustomerId`, `Surname`  
- Features include demographic, financial, and behavioural variables.  
- Churn rate is ~**20.37%**, so we evaluate using imbalance-aware metrics (Recall, F1, PR-AUC).

---

## Reproducible Pipeline (Leakage-Safe)

We implemented a leakage-safe machine learning pipeline:

**Preprocess → SMOTE → Model**

- **Preprocess:** `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- **SMOTE:** applied **inside** the pipeline so oversampling happens only on training folds
- **Split:** 70% train / 30% test (stratified)
- **Validation:** split from training only (used for threshold tuning); test set is untouched until final evaluation

---

## Models Compared

We compared four models under the **same pipeline** for fair evaluation:

- Logistic Regression (baseline)
- Decision Tree (interpretable)
- Random Forest (bagging ensemble; 300 trees)
- Gradient Boosting (best overall performance in our run)

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt

## RUN
bash
python churn_pipeline.py



## RUN THE PROGRAM

bash

python churn_pipeline.py


Outputs will be saved to report_figures/
