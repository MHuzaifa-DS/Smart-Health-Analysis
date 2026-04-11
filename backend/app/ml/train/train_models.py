"""
ml/train/train_models.py — Train all three disease prediction models.

Datasets (download first):
  - Diabetes:     https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  - Hypertension: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
  - Anemia:       https://www.kaggle.com/datasets/biswa96/anemia-detection

Run:
  python -m app.ml.train.train_models --disease all
  python -m app.ml.train.train_models --disease diabetes
"""
import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent / "data"


# ── Diabetes ────────────────────────────────────────────────────────────────────

def train_diabetes():
    """
    Train on Pima Indians Diabetes Database.
    Target: Outcome (0=no diabetes, 1=diabetes)
    """
    print("\n" + "="*60)
    print("TRAINING: Diabetes Model")
    print("="*60)

    data_path = DATA_DIR / "diabetes.csv"
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Download from: https://kaggle.com/uciml/pima-indians-diabetes-database")
        _create_synthetic_diabetes_data(data_path)

    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Feature columns
    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    target_col = "Outcome"

    # Handle zeros as missing (medically implausible zero values)
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train Random Forest (best for this dataset)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print(f"\nTest Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_prob):.4f}")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # Save
    metadata = {
        "model_name": "diabetes_random_forest",
        "version": "v1",
        "features": feature_cols,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "target_disease": "Type 2 Diabetes",
        "threshold": 0.5,
    }
    joblib.dump(rf, MODELS_DIR / "diabetes_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "diabetes_scaler.pkl")
    joblib.dump(metadata, MODELS_DIR / "diabetes_metadata.pkl")
    print(f"\n✅ Saved: diabetes_model.pkl, diabetes_scaler.pkl")
    return metadata


# ── Hypertension ────────────────────────────────────────────────────────────────

def train_hypertension():
    """
    Train on Framingham Heart Study dataset.
    Target: TenYearCHD (cardiovascular risk proxy for hypertension)
    """
    print("\n" + "="*60)
    print("TRAINING: Hypertension Model")
    print("="*60)

    data_path = DATA_DIR / "framingham.csv"
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        _create_synthetic_hypertension_data(data_path)

    df = pd.read_csv(data_path)
    df = df.dropna()
    print(f"Dataset shape (after dropna): {df.shape}")

    feature_cols = [
        "age", "male", "currentSmoker", "cigsPerDay",
        "BMI", "heartRate", "glucose", "sysBP",
        "totChol", "prevalentHyp"
    ]
    target_col = "TenYearCHD"

    # Rename to match our schema
    col_map = {
        "male": "gender_male",
        "currentSmoker": "smoking",
        "cigsPerDay": "cigs_per_day",
        "heartRate": "heart_rate",
        "sysBP": "systolic_bp",
        "totChol": "cholesterol",
        "prevalentHyp": "prevalent_hyp",
    }

    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient Boosting works well here
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 1]

    print(f"\nTest Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_prob):.4f}")
    cv_scores = cross_val_score(gb, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Risk", "High Risk"]))

    scaler = StandardScaler()
    scaler.fit(X_train)

    metadata = {
        "model_name": "hypertension_gradient_boosting",
        "version": "v1",
        "features": available_features,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "target_disease": "Hypertension / Cardiovascular Risk",
        "threshold": 0.4,  # lower threshold = higher sensitivity
    }
    joblib.dump(gb, MODELS_DIR / "hypertension_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "hypertension_scaler.pkl")
    joblib.dump(metadata, MODELS_DIR / "hypertension_metadata.pkl")
    print(f"\n✅ Saved: hypertension_model.pkl")
    return metadata


# ── Anemia ──────────────────────────────────────────────────────────────────────

def train_anemia():
    """
    Train on UCI Anemia Dataset.
    Target: Result (1=anemia, 0=normal)
    """
    print("\n" + "="*60)
    print("TRAINING: Anemia Model")
    print("="*60)

    data_path = DATA_DIR / "anemia.csv"
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        _create_synthetic_anemia_data(data_path)

    df = pd.read_csv(data_path)
    df = df.dropna()
    print(f"Dataset shape: {df.shape}")

    # Try to find target column
    target_col = next(
        (c for c in ["Result", "Anaemic", "anemia", "label", "target"] if c in df.columns),
        None
    )
    if not target_col:
        raise ValueError(f"No target column found. Columns: {df.columns.tolist()}")

    feature_cols = [c for c in ["Hemoglobin", "MCH", "MCHC", "MCV"] if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SVM with RBF kernel works well for anemia
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced")
    svm.fit(X_train_s, y_train)

    y_pred = svm.predict(X_test_s)
    y_prob = svm.predict_proba(X_test_s)[:, 1]

    print(f"\nTest Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_prob):.4f}")
    cv_scores = cross_val_score(svm, X_train_s, y_train, cv=5, scoring="accuracy")
    print(f"5-Fold CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anemia"]))

    metadata = {
        "model_name": "anemia_svm",
        "version": "v1",
        "features": feature_cols,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "target_disease": "Anemia",
        "threshold": 0.5,
        "requires_scaling": True,
    }
    joblib.dump(svm, MODELS_DIR / "anemia_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "anemia_scaler.pkl")
    joblib.dump(metadata, MODELS_DIR / "anemia_metadata.pkl")
    print(f"\n✅ Saved: anemia_model.pkl")
    return metadata


# ── Synthetic data generators (fallback when real datasets unavailable) ─────────

def _create_synthetic_diabetes_data(path: Path):
    """Generate realistic synthetic diabetes data for development/testing."""
    print("⚠️  Creating synthetic diabetes data for development...")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 768

    # Approximate Pima Indians distribution
    data = {
        "Pregnancies": np.random.poisson(3.8, n).clip(0, 17),
        "Glucose": np.where(
            np.random.random(n) < 0.35,
            np.random.normal(155, 20, n),   # diabetic
            np.random.normal(109, 18, n),   # non-diabetic
        ).clip(44, 199),
        "BloodPressure": np.random.normal(69, 19, n).clip(0, 122),
        "SkinThickness": np.random.normal(20, 16, n).clip(0, 99),
        "Insulin": np.random.exponential(80, n).clip(0, 846),
        "BMI": np.random.normal(32, 7, n).clip(0, 67),
        "DiabetesPedigreeFunction": np.random.exponential(0.47, n).clip(0.08, 2.42),
        "Age": np.random.normal(33, 11, n).clip(21, 81).astype(int),
    }
    df = pd.DataFrame(data)
    df["Outcome"] = (
        (df["Glucose"] > 140).astype(int) |
        (df["BMI"] > 35).astype(int) & (df["Age"] > 40).astype(int)
    ).clip(0, 1)
    df.to_csv(path, index=False)
    print(f"   Synthetic data saved to {path} ({n} rows)")


def _create_synthetic_hypertension_data(path: Path):
    print("⚠️  Creating synthetic hypertension data for development...")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 4240
    data = {
        "age": np.random.normal(49, 8, n).clip(32, 70).astype(int),
        "male": np.random.binomial(1, 0.43, n),
        "currentSmoker": np.random.binomial(1, 0.49, n),
        "cigsPerDay": np.random.poisson(9, n).clip(0, 70),
        "BMI": np.random.normal(25.8, 4, n).clip(15, 56),
        "heartRate": np.random.normal(75, 12, n).clip(44, 143).astype(int),
        "glucose": np.random.normal(82, 24, n).clip(40, 394),
        "sysBP": np.random.normal(132, 22, n).clip(83, 295),
        "totChol": np.random.normal(236, 44, n).clip(107, 696),
        "prevalentHyp": np.random.binomial(1, 0.31, n),
    }
    df = pd.DataFrame(data)
    df["TenYearCHD"] = (
        (df["sysBP"] > 160).astype(int) |
        (df["age"] > 60).astype(int) & (df["currentSmoker"] == 1).astype(int)
    ).clip(0, 1)
    df.to_csv(path, index=False)
    print(f"   Synthetic data saved to {path}")


def _create_synthetic_anemia_data(path: Path):
    print("⚠️  Creating synthetic anemia data for development...")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 1000
    anemia_mask = np.random.random(n) < 0.4
    data = {
        "Hemoglobin": np.where(anemia_mask, np.random.normal(9.5, 1.5, n), np.random.normal(14.0, 1.5, n)).clip(5, 20),
        "MCH": np.where(anemia_mask, np.random.normal(24, 3, n), np.random.normal(29, 2, n)).clip(15, 40),
        "MCHC": np.where(anemia_mask, np.random.normal(30, 2, n), np.random.normal(34, 1.5, n)).clip(25, 40),
        "MCV": np.where(anemia_mask, np.random.normal(72, 8, n), np.random.normal(88, 6, n)).clip(50, 120),
        "Result": anemia_mask.astype(int),
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"   Synthetic data saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=["diabetes", "hypertension", "anemia", "all"], default="all")
    args = parser.parse_args()

    results = {}
    if args.disease in ("diabetes", "all"):
        results["diabetes"] = train_diabetes()
    if args.disease in ("hypertension", "all"):
        results["hypertension"] = train_hypertension()
    if args.disease in ("anemia", "all"):
        results["anemia"] = train_anemia()

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for disease, meta in results.items():
        print(f"  {disease:15s}: accuracy={meta['accuracy']:.4f}, auc={meta['auc']:.4f}")
    print("="*60)
