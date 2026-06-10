from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def load_data(
    train_path: str,
    test_path: str,
    contract_path: str,
    target_col: str = "target"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Ingest feature contract and return deterministic partitioned Pandas DataFrames."""
    df_contract = pl.read_csv(contract_path)
    approved_features = df_contract.filter(pl.col("status") == "KEEP")["Feature_Name"].to_list()
    approved_features.sort()

    df_train_pl = pl.scan_parquet(train_path).select([*approved_features, target_col]).collect()
    df_test_pl = pl.scan_parquet(test_path).select([*approved_features, target_col]).collect()

    df_train = df_train_pl.to_pandas()
    df_test = df_test_pl.to_pandas()

    X_train = df_train[approved_features]
    y_train = df_train[target_col]
    X_test = df_test[approved_features]
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test, approved_features

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Train synchronous baseline and champion classifiers."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, tree_method="hist")

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return {
        "Random_Forest_Baseline": rf,
        "XGBoost_Champion": xgb
    }

def find_optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes the macro F1-Score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return float(best_threshold)

def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, model_name: str, preset_threshold: float = None) -> Dict[str, Any]:
    """Evaluate classifier using strict institutional metrics including Gini and dynamic thresholding."""
    y_prob = model.predict_proba(X)[:, 1]
    if preset_threshold is not None:
        optimal_threshold = preset_threshold
    else:
        optimal_threshold = find_optimal_threshold(y, y_prob)
    y_pred = (y_prob >= optimal_threshold).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    gini = 2 * roc_auc - 1
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    metrics = {
        "Model": model_name,
        "ROC-AUC": roc_auc,
        "Gini": gini,
        "PR-AUC": pr_auc,
        "Optimal-Threshold": optimal_threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "y_prob": y_prob
    }
    return metrics

def build_comparison_matrix(metrics_list: List[Dict[str, Any]], save_path: str) -> None:
    """Compile metrics into a Pandas DataFrame ledger and dump to CSV."""
    records = []
    for m in metrics_list:
        rec = {k: v for k, v in m.items() if k != "y_prob"}
        records.append(rec)
    
    df = pd.DataFrame(records)
    print("============================================================")
    print("MODEL COMPARISON MATRIX")
    print("============================================================")
    print(df.to_string(index=False))
    print("============================================================")
    
    df.to_csv(save_path, index=False)
    print(f"[EXPORT] Comparison matrix saved -> {save_path}")

def plot_evaluation_curves(model_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save_dir: str) -> None:
    """Plot ROC and PR curves aggressively sealing memory instances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    for name, model in model_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f"{name} AUC={auc_val:.4f}")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc_val = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, label=f"{name} PR-AUC={pr_auc_val:.4f}")

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    plt.tight_layout()
    save_path = Path(save_dir) / "model_evaluation_curves.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Evaluation curves saved -> {save_path}")

def plot_confusion_matrices(
    model_dict: Dict[str, Any], 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    save_dir: str,
    frozen_thresholds: Dict[str, float] = None
) -> None:
    """Plot confusion matrices using optimal dynamic thresholds for each model."""
    num_models = len(model_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), dpi=150)
    
    # Ensure axes is iterable even if there is only 1 model
    if num_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, model_dict.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        opt_thresh = frozen_thresholds.get(name, 0.5) if frozen_thresholds is not None else 0.5
        y_pred = (y_prob >= opt_thresh).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{name}\nFrozen Threshold: {opt_thresh:.2f}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    save_path = Path(save_dir) / "confusion_matrices.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Confusion matrices saved -> {save_path}")

def plot_shap_summary(model: Any, X_train: pd.DataFrame, feature_names: List[str], save_path: str, max_display_samples: int = 1000) -> None:
    """Generate high-dimensional SHAP beeswarm summary."""
    X_sample = X_train.sample(n=min(len(X_train), max_display_samples), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=15, show=False)
    plt.title("SHAP Summary Plot")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] SHAP summary plot saved -> {save_path}")

def save_artifacts(model_dict: Dict[str, Any], X_train: pd.DataFrame, save_dir: str) -> None:
    """Serialize models and execute a cold validation verification gate."""
    save_path_dir = Path(save_dir)
    save_path_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in model_dict.items():
        path = save_path_dir / f"{name.lower()}.joblib"
        joblib.dump(model, path)
        print(f"[SAVE] Model {name} saved to {path}")
        
        # Cold Validation Gate
        print(f"[VALIDATE] Executing cold-reload validation for {name}...")
        loaded_model = joblib.load(path)
        dummy_row = X_train.iloc[[0]]
        
        prob = loaded_model.predict_proba(dummy_row)
        assert prob.shape == (1, 2), f"Assertion Error: Cold validation structural footprint mismatch for {name}. Shape: {prob.shape}"
        
        print(f"[VALIDATE] Cold-reload validation passed successfully for {name}.")
