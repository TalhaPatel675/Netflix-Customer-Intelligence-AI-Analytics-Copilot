"""
Machine Learning — train the 3 PRD models.
==========================================
Model 1 — Churn Prediction (classification)
Model 2 — Customer Segmentation (K-Means)
Model 3 — Customer Lifetime Value (regression)

Saves trained artifacts to models/:
  churn_model.pkl, churn_metrics.json, scaler.pkl, encoder.pkl
  segment_model.pkl, segment_labels.json
  clv_model.pkl, clv_metrics.json
  feature_columns.json
"""
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "customer_tenure_days", "avg_watch_time_per_session", "monthly_watch_time_trend",
    "days_since_last_activity", "login_frequency", "support_ticket_count",
    "avg_customer_satisfaction", "failed_payment_count", "failed_payment_rate",
    "plan_changes", "engagement_score", "customer_lifetime_value",
    "total_watch_minutes", "avg_completion_percentage", "distinct_titles_watched",
    "distinct_devices", "avg_session_duration", "negative_feedback_count",
    "avg_feedback_rating", "current_plan_tier",
    "age",
]
CATEGORICAL = ["gender", "country", "acquisition_channel", "customer_segment", "preferred_language"]


def load_data():
    df = pd.read_csv(PROC / "customer_features.csv")
    return df


def prep_features(df):
    X = df[FEATURES].copy()
    # categorical encoding
    encoders = {}
    for c in CATEGORICAL:
        le = LabelEncoder()
        X[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    return X, encoders


# ---------------------------------------------------------------------------
# MODEL 1 — Churn classification
# ---------------------------------------------------------------------------
def train_churn(df):
    X, encoders = prep_features(df)
    y = df["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss",
                                 scale_pos_weight=(y == 0).sum() / (y == 1).sum()),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "cv_auc_mean": float(cv.mean()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        print(f"  {name:22s} AUC={results[name]['roc_auc']:.4f}  "
              f"P={results[name]['precision']:.3f} R={results[name]['recall']:.3f} "
              f"F1={results[name]['f1']:.3f}")

    # pick best by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = models[best_name]
    best.fit(X_train_s, y_train)

    # feature importances (tree models) or coefficients
    if hasattr(best, "feature_importances_"):
        imp = dict(zip(X.columns, best.feature_importances_))
    else:
        imp = dict(zip(X.columns, np.abs(best.coef_[0])))
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    with open(MODEL_DIR / "churn_model.pkl", "wb") as f:
        pickle.dump({"model": best, "scaler": scaler, "encoders": encoders,
                     "features": FEATURES, "categorical": CATEGORICAL}, f)
    with open(MODEL_DIR / "churn_metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results,
                   "feature_importance": imp}, f, indent=2)
    print(f"  -> Best churn model: {best_name} (AUC={results[best_name]['roc_auc']:.4f})")
    return best, scaler, encoders


# ---------------------------------------------------------------------------
# MODEL 2 — Customer segmentation (K-Means)
# ---------------------------------------------------------------------------
def train_segmentation(df):
    seg_features = ["engagement_score", "customer_lifetime_value",
                    "customer_tenure_days", "avg_customer_satisfaction",
                    "failed_payment_rate", "avg_watch_time_per_session"]
    X = df[seg_features].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Elbow + silhouette
    Ks = range(2, 9)
    inertias, sil_scores = [], []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(Xs, km.labels_))
    best_k = Ks[int(np.argmax(sil_scores))]

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    # business naming from cluster means
    seg_df = X.copy()
    seg_df["cluster"] = labels
    profile = seg_df.groupby("cluster")[seg_features].mean()
    names = {}
    for cl in profile.index:
        row = profile.loc[cl]
        if row["customer_lifetime_value"] > profile["customer_lifetime_value"].median() and \
           row["engagement_score"] > profile["engagement_score"].median():
            names[int(cl)] = "High-Value Loyal"
        elif row["failed_payment_rate"] > profile["failed_payment_rate"].median() and \
             row["avg_customer_satisfaction"] < profile["avg_customer_satisfaction"].median():
            names[int(cl)] = "At-Risk Low-Engagement"
        elif row["customer_tenure_days"] < profile["customer_tenure_days"].median():
            names[int(cl)] = "Price-Sensitive New Signups"
        else:
            names[int(cl)] = "Moderate Engaged"

    with open(MODEL_DIR / "segment_model.pkl", "wb") as f:
        pickle.dump({"model": km, "scaler": scaler, "features": seg_features}, f)
    with open(MODEL_DIR / "segment_labels.json", "w") as f:
        json.dump({"best_k": int(best_k), "names": names, "silhouette": float(max(sil_scores)),
                   "elbow_inertias": inertias, "silhouette_scores": sil_scores}, f, indent=2)
    print(f"  -> Segmentation: K={best_k}, silhouette={max(sil_scores):.3f}")
    for cl, nm in names.items():
        print(f"     cluster {cl}: {nm}")
    return km, scaler


# ---------------------------------------------------------------------------
# MODEL 3 — CLV regression
# ---------------------------------------------------------------------------
def train_clv(df):
    reg_features = [f for f in FEATURES if f not in ["customer_lifetime_value", "churned"]]
    X = df[reg_features].copy()
    for c in CATEGORICAL:
        X[c] = LabelEncoder().fit_transform(df[c].astype(str))
    y = df["customer_lifetime_value"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        results[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }
        print(f"  {name:22s} MAE={results[name]['mae']:.2f} RMSE={results[name]['rmse']:.2f} "
              f"R2={results[name]['r2']:.3f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    best = models[best_name]
    best.fit(X_train_s, y_train)

    with open(MODEL_DIR / "clv_model.pkl", "wb") as f:
        pickle.dump({"model": best, "scaler": scaler, "features": reg_features}, f)
    with open(MODEL_DIR / "clv_metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"  -> Best CLV model: {best_name} (R2={results[best_name]['r2']:.3f})")
    return best


def main():
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    df = load_data()
    print(f"Dataset: {df.shape[0]} customers x {df.shape[1]} features\n")

    print("[Model 1] Churn classification")
    train_churn(df)
    print("\n[Model 2] Customer segmentation")
    train_segmentation(df)
    print("\n[Model 3] CLV regression")
    train_clv(df)
    print("\nAll models saved to models/")


if __name__ == "__main__":
    main()
