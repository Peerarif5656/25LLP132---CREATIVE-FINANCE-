# Predicting Bank Customer Churn Using Machine Learning

**Module:** 25LLP132 – Principles of Artificial Intelligence and Data Analytics  
**Group:** Creative Finance  
**Project:** Predicting Bank Customer Churn Using Machine Learning  
**Dataset:** Kaggle “Churn Modelling” (`Churn_Modelling.csv`)

This repository contains the reproducible pipeline used in our project .

## Business Context
Customer churn is costly: retaining customers is usually cheaper than acquiring new ones.  
Our goal is to identify high-risk customers early so the bank can apply targeted retention strategies.

## Dataset Summary
- Rows: **10,000 customers**
- Target: **Exited** (1 = churned, 0 = retained)
- Removed non-predictive identifiers: **RowNumber, CustomerId, Surname**
- Features include demographic, financial, and behavioural variables
- Churn rate: **~20.37%**, so evaluation focuses on **Recall, F1, and PR-AUC** (not accuracy alone)

## Leakage-Safe Pipeline (Key Design)
**Preprocess → SMOTE → Model**

- Preprocess: `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- SMOTE is applied **inside the pipeline**, so oversampling happens only on training folds (prevents leakage)
- Split: **70% train / 30% test** (stratified)
- Validation: split from training only (used for threshold exploration); test set untouched until final evaluation

## Models Compared
- Logistic Regression (baseline)
- Decision Tree (interpretable)
- Random Forest (300 trees)
- Gradient Boosting (best overall performance in our run)

## How to Run

### 1) Place the dataset
Download `Churn_Modelling.csv` from Kaggle and place it in the same folder as `churn_pipeline.py`.

> Note: The dataset is not included in this repository to respect Kaggle licensing.

### 2) Install dependencies
```bash
pip install -r requirements.txt

### 3) Run
python churn_pipeline.py
