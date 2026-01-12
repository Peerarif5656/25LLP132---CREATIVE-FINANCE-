"""
25LLP132 – Principles of Artificial Intelligence and Data Analytics
Group: Creative Finance

Project: Predicting Bank Customer Churn Using Machine Learning
Dataset: Kaggle “Churn Modelling” (Churn_Modelling.csv)

This script reproduces are :
- dataset summary tables
- EDA figures + key stats
- leakage-safe modelling pipeline (Preprocess → SMOTE → Model)
- test-set evaluation + 5-fold stratified cross-validation
- best-model diagnostics (confusion matrix, ROC, PR curve, threshold exploration)
- actionable business output: Top 5% high-risk customer ranking

Outputs are saved into: report_figures/
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Safe for marker environments (no GUI required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_curve,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


#  SETTINGS (match report + reproducibility)

CSV_PATH = "Churn_Modelling.csv" # Put this file in the same folder as this script
TARGET = "Exited"
OUT_DIR = "report_figures"

RANDOM_STATE = 42
TEST_SIZE = 0.30 # Train 70% / Test 30%
VAL_SIZE_FROM_TRAIN = 0.20 # Validation subset from training only (threshold exploration)
SMOTE_K = 3 # More stable for CV folds

SHOW_PLOTS = False # Keep False for submission

os.makedirs(OUT_DIR, exist_ok=True)


def savefig(filename: str) -> None:
    """Save current matplotlib figure into OUT_DIR with high resolution."""
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)


def maybe_show() -> None:
    """Show plots only when SHOW_PLOTS is enabled; always close for clean runs."""
    if SHOW_PLOTS:
        plt.show()
    plt.close()


#  LOAD DATA (safe checks + consistent naming)

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cannot find '{csv_path}'. Place Churn_Modelling.csv in the SAME folder as churn_pipeline.py."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found. Columns: {df.columns.tolist()}")

    # Drop common identifier columns (Kaggle version)
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df[TARGET] = df[TARGET].astype(int)
    return df

#  REPORT TABLES

def make_dataset_tables(df: pd.DataFrame) -> Dict:
    table0 = pd.DataFrame(
        {
            "Feature": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique": df.nunique().values,
        }
    )
    table0.to_csv(os.path.join(OUT_DIR, "Table_0_Data_Summary.csv"), index=False)

    stats: Dict = {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Churn_%": round(df[TARGET].mean() * 100, 2),
        "Retain_%": round((1 - df[TARGET].mean()) * 100, 2),
    }

    if "Age" in df.columns:
        stats["Median_Age_Retained"] = float(df.loc[df[TARGET] == 0, "Age"].median())
        stats["Median_Age_Churned"] = float(df.loc[df[TARGET] == 1, "Age"].median())

    pd.DataFrame([stats]).to_csv(os.path.join(OUT_DIR, "Table_Stats_For_Report.csv"), index=False)
    print("Saved: Table_0_Data_Summary.csv, Table_Stats_For_Report.csv")
    return stats



#  SPLITS: train/test + validation from training only

def make_splits(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X, y, X_train_full, X_test, y_train_full, y_test, X_train, X_val, y_train, y_val


#  PREPROCESS + PIPELINE (preprocess → SMOTE → model)

def make_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    print("Categorical features:", categorical_features)
    print("Numeric features:", numeric_features)
    return preprocess, categorical_features, numeric_features


def build_pipeline(preprocess: ColumnTransformer, model) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)),
            ("model", model),
        ]
    )


#  EDA FIGURES + EDA SUMMARY TABLE

def run_eda(df: pd.DataFrame, stats: Dict) -> None:
    # Figure 3: target distribution
    plt.figure()
    (df[TARGET].value_counts(normalize=True) * 100).sort_index().plot(kind="bar")
    plt.title("Target Distribution (0=Retained, 1=Churned)")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    savefig("Figure_3_Target_Distribution.png")
    maybe_show()

    # Figure 2 + Figure 4: categorical distribution + churn rate by category
    cat_cols = [c for c in ["Geography", "Gender", "IsActiveMember", "HasCrCard"] if c in df.columns]
    if cat_cols:
        n = len(cat_cols)
        cols = 2
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(10, 4 * rows))
        for i, col in enumerate(cat_cols, 1):
            ax = plt.subplot(rows, cols, i)
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"{col} Distribution")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
        savefig("Figure_2_Categorical_Distribution.png")
        maybe_show()

        plt.figure(figsize=(10, 4 * rows))
        for i, col in enumerate(cat_cols, 1):
            ax = plt.subplot(rows, cols, i)
            df.groupby(col)[TARGET].mean().sort_values(ascending=False).plot(kind="bar", ax=ax)
            ax.set_title(f"Churn Rate by {col}")
            ax.set_ylabel("Churn Rate")
            plt.xticks(rotation=45, ha="right")
        savefig("Figure_4_Exited_Analysis.png")
        maybe_show()

    # Figure 5: boxplots
    box_cols = [c for c in ["Age", "CreditScore", "Balance", "EstimatedSalary"] if c in df.columns]
    if box_cols:
        n = len(box_cols)
        cols = 2
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(10, 4 * rows))
        for i, col in enumerate(box_cols, 1):
            ax = plt.subplot(rows, cols, i)
            df.boxplot(column=col, by=TARGET, ax=ax)
            ax.set_title(f"{col} by Exited")
            ax.set_xlabel("Exited")
            ax.set_ylabel(col)
        plt.suptitle("")
        savefig("Figure_5_Boxplots.png")
        maybe_show()

    # Figure 1: correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.title("Feature Correlation Matrix for Customer Churn Analysis")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    savefig("Figure_1_Correlation_Heatmap.png")
    maybe_show()

    # Table_EDA_KeyStats.csv
    eda_stats = [
        {"Metric": "Churn_%", "Value": stats["Churn_%"]},
        {"Metric": "Retain_%", "Value": stats["Retain_%"]},
    ]

    if "Age" in df.columns:
        eda_stats += [
            {"Metric": "Median_Age_Retained", "Value": stats.get("Median_Age_Retained", np.nan)},
            {"Metric": "Median_Age_Churned", "Value": stats.get("Median_Age_Churned", np.nan)},
        ]

    for col in ["Geography", "Gender", "IsActiveMember", "HasCrCard"]:
        if col in df.columns:
            rates = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
            for k, v in rates.head(3).items():
                eda_stats.append({"Metric": f"ChurnRate_{col}_{k}", "Value": round(float(v), 3)})

    pd.DataFrame(eda_stats).to_csv(os.path.join(OUT_DIR, "Table_EDA_KeyStats.csv"), index=False)
    print("Saved: Table_EDA_KeyStats.csv")


#  MODELS

def get_models() -> Dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_estimators=300
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


#  TEST EVALUATION (Table 1)

def evaluate_on_test(models: Dict, preprocess: ColumnTransformer,
                     X_train_full, y_train_full, X_test, y_test):
    results = []

    for name, model in models.items():
        pipe = build_pipeline(preprocess, model)
        pipe.fit(X_train_full, y_train_full)

        pred = pipe.predict(X_test)

        auc = np.nan
        ap = np.nan
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, prob)
                ap = average_precision_score(y_test, prob)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": auc,
            "PR_AUC_AP": ap,
        })

    results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)

    out = results_df.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "AUC", "PR_AUC_AP"]:
        out[c] = out[c].astype(float).round(3)

    out.to_csv(os.path.join(OUT_DIR, "Table_1_Model_Performance_TestSet.csv"), index=False)
    print("Saved: Table_1_Model_Performance_TestSet.csv")
    return results_df


#  5-FOLD STRATIFIED CV (Table 2 + Figure 10)

def evaluate_with_cv(models: Dict, preprocess: ColumnTransformer, X_train_full, y_train_full):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []

    for name, model in models.items():
        pipe = build_pipeline(preprocess, model)

        scores = cross_validate(
            pipe,
            X_train_full,
            y_train_full,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )

        cv_rows.append({
            "Model": name,
            "CV_Accuracy_mean": scores["test_accuracy"].mean(),
            "CV_Accuracy_std": scores["test_accuracy"].std(),
            "CV_Precision_mean": scores["test_precision"].mean(),
            "CV_Precision_std": scores["test_precision"].std(),
            "CV_Recall_mean": scores["test_recall"].mean(),
            "CV_Recall_std": scores["test_recall"].std(),
            "CV_F1_mean": scores["test_f1"].mean(),
            "CV_F1_std": scores["test_f1"].std(),
            "CV_AUC_mean": scores["test_roc_auc"].mean(),
            "CV_AUC_std": scores["test_roc_auc"].std(),
        })

    cv_df = pd.DataFrame(cv_rows).sort_values(by="CV_F1_mean", ascending=False)

    out = cv_df.copy()
    for c in out.columns:
        if c != "Model":
            out[c] = out[c].astype(float).round(3)

    out.to_csv(os.path.join(OUT_DIR, "Table_2_CrossValidation_5Fold.csv"), index=False)
    print("Saved: Table_2_CrossValidation_5Fold.csv")

    plt.figure()
    order = out.sort_values("CV_F1_mean", ascending=True)
    plt.errorbar(order["CV_F1_mean"], order["Model"], xerr=order["CV_F1_std"], fmt="o")
    plt.title("Cross-Validation F1 (mean ± std)")
    plt.xlabel("F1")
    savefig("Figure_10_CV_F1_ErrorBars.png")
    maybe_show()

    return out



#  BEST MODEL DIAGNOSTICS + THRESHOLD EXPLORATION

def best_model_diagnostics(best_name: str, models: Dict, preprocess: ColumnTransformer,
                           X_train_full, y_train_full, X_test, y_test,
                           X_train, y_train, X_val, y_val,
                           categorical_features: List[str], numeric_features: List[str]) -> ImbPipeline:
    best_pipe = build_pipeline(preprocess, models[best_name])

    # Threshold exploration (validation only)
    best_pipe.fit(X_train, y_train)
    best_threshold = 0.5

    if hasattr(best_pipe, "predict_proba"):
        val_prob = best_pipe.predict_proba(X_val)[:, 1]
        prec, rec, thr = precision_recall_curve(y_val, val_prob)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        idx = int(np.argmax(f1s[:-1])) if len(thr) > 0 else None
        best_threshold = float(thr[idx]) if idx is not None else 0.5

    # Refit on full training (no test leakage)
    best_pipe.fit(X_train_full, y_train_full)

    test_pred_default = best_pipe.predict(X_test)
    test_prob = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, "predict_proba") else None

    test_pred_tuned = test_pred_default
    if test_prob is not None:
        test_pred_tuned = (test_prob >= best_threshold).astype(int)

    # Figure 6: confusion matrix
    cm = confusion_matrix(y_test, test_pred_default)
    plt.figure()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f"Confusion Matrix (threshold=0.50) - {best_name}")
    savefig("Figure_6_Confusion_Matrix.png")
    maybe_show()

    cm_table = pd.DataFrame(
        {"Pred_0": [cm[0, 0], cm[1, 0]], "Pred_1": [cm[0, 1], cm[1, 1]]},
        index=["True_0", "True_1"],
    )
    cm_table.to_csv(os.path.join(OUT_DIR, "Table_Confusion_Matrix_Values.csv"), index=True)
    print("Saved: Table_Confusion_Matrix_Values.csv")

    # Figure 7 + Figure 9
    if test_prob is not None and len(np.unique(y_test)) == 2:
        auc_val = roc_auc_score(y_test, test_prob)

        plt.figure()
        RocCurveDisplay.from_predictions(y_test, test_prob)
        plt.title(f"ROC Curve - {best_name} (AUC={auc_val:.3f})")
        savefig("Figure_7_ROC_Curve.png")
        maybe_show()

        plt.figure()
        PrecisionRecallDisplay.from_predictions(y_test, test_prob)
        plt.title(f"Precision-Recall Curve - {best_name}")
        savefig("Figure_9_Precision_Recall_Curve.png")
        maybe_show()

    # Table 3: threshold exploration summary
    table3 = pd.DataFrame([{
        "Best_Model": best_name,
        "Threshold_Selected_on_Validation": round(best_threshold, 3),
        "Test_Recall_default_0.50": round(recall_score(y_test, test_pred_default, zero_division=0), 3),
        "Test_F1_default_0.50": round(f1_score(y_test, test_pred_default, zero_division=0), 3),
        "Test_Recall_tuned": round(recall_score(y_test, test_pred_tuned, zero_division=0), 3),
        "Test_F1_tuned": round(f1_score(y_test, test_pred_tuned, zero_division=0), 3),
    }])
    table3.to_csv(os.path.join(OUT_DIR, "Table_3_Threshold_Tuning.csv"), index=False)
    print("Saved: Table_3_Threshold_Tuning.csv")

    # Business output: Top 5% high-risk customers
    if test_prob is not None:
        risk_out = X_test.copy()
        risk_out["Churn_Probability"] = test_prob
        risk_out["True_Exited"] = y_test.values
        risk_out["Pred_Default"] = test_pred_default

        top_n = max(1, int(0.05 * len(risk_out)))
        top_risk = risk_out.sort_values("Churn_Probability", ascending=False).head(top_n)
        top_risk.to_csv(os.path.join(OUT_DIR, "Top_5pct_HighRisk_Customers.csv"), index=False)
        print("Saved: Top_5pct_HighRisk_Customers.csv")

    # Feature importance (Figure 8 + CSV) for tree-based models
    if best_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist() if categorical_features else []
        feature_names = numeric_features + cat_names

        fitted_model = best_pipe.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": fitted_model.feature_importances_,
            }).sort_values("importance", ascending=False).head(15)

            plt.figure(figsize=(8, 6))
            plt.barh(fi["feature"][::-1], fi["importance"][::-1])
            plt.title(f"Top 15 Feature Importances - {best_name}")
            savefig("Figure_8_Feature_Importance.png")
            maybe_show()

            fi.to_csv(os.path.join(OUT_DIR, "Feature_Importance_Top15.csv"), index=False)
            print("Saved: Feature_Importance_Top15.csv")

    return best_pipe


#  OPTIONAL FAIRNESS CHECK (Recall by group)

def fairness_recall_table(pipe: ImbPipeline, X_te, y_te, group_col: str) -> pd.DataFrame:
    temp = X_te[[group_col]].copy()
    temp["y_true"] = y_te.values
    temp["y_pred"] = pipe.predict(X_te)

    out = []
    for g, sub in temp.groupby(group_col, dropna=False):
        r = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)
        out.append({"Group": g, "Recall": r, "Count": int(len(sub))})
    return pd.DataFrame(out).sort_values("Recall", ascending=False)

def main() -> None:
    df = load_data(CSV_PATH)
    print("Dataset shape:", df.shape)

    stats = make_dataset_tables(df)

    X, y, X_train_full, X_test, y_train_full, y_test, X_train, X_val, y_train, y_val = make_splits(df)
    preprocess, categorical_features, numeric_features = make_preprocess(X)

    run_eda(df, stats)

    models = get_models()

    results_df = evaluate_on_test(models, preprocess, X_train_full, y_train_full, X_test, y_test)
    _ = evaluate_with_cv(models, preprocess, X_train_full, y_train_full)

    best_name = results_df.iloc[0]["Model"]
    print("Best model (by Test F1):", best_name)

    best_pipe = best_model_diagnostics(
        best_name, models, preprocess,
        X_train_full, y_train_full, X_test, y_test,
        X_train, y_train, X_val, y_val,
        categorical_features, numeric_features,
    )

for col in ["Gender", "Geography"]:
        if col in X_test.columns:
            ft = fairness_recall_table(best_pipe, X_test, y_test, col)
            ft.to_csv(os.path.join(OUT_DIR, f"Table_Fairness_Recall_by_{col}.csv"), index=False)
            print(f"Saved: Table_Fairness_Recall_by_{col}.csv")

    print("\n FINISHED. All outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
