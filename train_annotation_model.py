"""
train_annotation_model.py
--------------------------
Train a Random Forest classifier on annotation-based features (variant type,
molecular consequence) extracted from ClinVar HBB records.

Usage:
    python src/train_annotation_model.py --data data/clinvar_result.txt
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from preprocess import (
    encode_features,
    filter_hbb,
    get_feature_matrix,
    load_clinvar,
    make_binary_label,
)

FEATURE_COLS = ["Variant type", "Molecular consequence"]
RESULTS_DIR = "results"


def oversample(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Simple random oversampling of the minority class."""
    rng = np.random.RandomState(random_state)
    minority = y[y == 1].index if y.mean() < 0.5 else y[y == 0].index
    majority = y[y == 0].index if y.mean() < 0.5 else y[y == 1].index

    n_to_add = len(majority) - len(minority)
    minority_idx = rng.choice(minority, size=n_to_add, replace=True)
    X_bal = pd.concat([X, X.loc[minority_idx]])
    y_bal = pd.concat([y, y.loc[minority_idx]])
    return X_bal.reset_index(drop=True), y_bal.reset_index(drop=True)


def train(data_path: str, output_dir: str = RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # --- Data prep ---
    raw = load_clinvar(data_path)
    hbb = filter_hbb(raw)
    labelled = make_binary_label(hbb)
    encoded = encode_features(labelled, FEATURE_COLS)
    X, y = get_feature_matrix(encoded)

    print(f"Dataset: {len(X)} variants | Pathogenic: {y.sum()} | Benign: {(y==0).sum()}")

    X_bal, y_bal = oversample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # --- Model ---
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy : {report['accuracy']:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))

    # --- Save metrics ---
    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc": auc,
        "precision_pathogenic": report["1"]["precision"],
        "recall_pathogenic": report["1"]["recall"],
        "f1_pathogenic": report["1"]["f1-score"],
    }
    with open(os.path.join(output_dir, "annotation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve – Annotation Model")
    axes[0].legend()

    # --- Feature importance ---
    importances = pd.Series(clf.feature_importances_, index=X.columns).nlargest(15)
    # Clean up column names for display
    importances.index = importances.index.str.replace(
        r"^(Variant type_|Molecular consequence_)", "", regex=True
    )
    importances.sort_values().plot(kind="barh", ax=axes[1], color="#2563EB")
    axes[1].set_title("Top 15 Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "annotation_model_plots.png"), dpi=150)
    print(f"\nPlots saved to {output_dir}/annotation_model_plots.png")

    # --- Confusion matrix ---
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Benign", "Pathogenic"], ax=ax2
    )
    ax2.set_title("Confusion Matrix – Annotation Model")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "annotation_confusion_matrix.png"), dpi=150)

    return clf, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/clinvar_result.txt")
    parser.add_argument("--output", default=RESULTS_DIR)
    args = parser.parse_args()
    train(args.data, args.output)
