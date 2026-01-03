# Predicting Bank Customer Churn Using Machine Learning

**Group:** Creative Finance  
**Module:** 25LLP132 – Principles of Artificial Intelligence and Data Analytics  
**Project:** Predicting Bank Customer Churn Using Machine Learning  
**Dataset:** Kaggle “Churn Modelling” (`Churn_Modelling.csv`)

This repository contains the **reproducible leakage-safe pipeline** used in our **executive report** and **video presentation**.

---

## Project Goal (Business Context)

Customer churn is costly: retaining customers is usually cheaper than acquiring new ones.  
Our goal is to **identify high-risk customers early** so the bank can apply **targeted retention** rather than contacting all customers.

---

## Dataset Summary

- Rows: **10,000** customers  
- Target: **Exited** (1 = churned, 0 = retained)  
- Removed non-predictive identifiers: **RowNumber, CustomerId, Surname**  
- Features include **demographic**, **financial**, and **behavioural** variables  
- Churn rate is ~**20.37%**, so we evaluate using **imbalance-aware** metrics (**Recall, F1, PR-AUC**)

---

## Reproducible Pipeline (Leakage-Safe)

We implement the same pipeline for each model:

**Preprocess → SMOTE → Model**

- Preprocess:
  - `StandardScaler` (numeric)
  - `OneHotEncoder(handle_unknown="ignore")` (categorical)
- SMOTE:
  - applied **inside the pipeline**, so oversampling happens **only on training folds**
  - **no leakage** into validation or test data
- Split:
  - **70% train / 30% test** (stratified)
- Validation:
  - split from training only (used for threshold tuning); test set untouched until final evaluation

---

## Models Compared

All models run under the same pipeline for a fair comparison:

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

