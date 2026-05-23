import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from src.config import REPORTS_DIR


def plot_confusion_matrix(y_test, y_pred, reports_dir: str = REPORTS_DIR) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    path = str(Path(reports_dir) / "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(y_test, y_prob, reports_dir: str = REPORTS_DIR) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    path = str(Path(reports_dir) / "roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(estimator, feature_names: list, reports_dir: str = REPORTS_DIR) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    importance = None
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    if importance is None:
        return ""
    top_n = min(10, len(feature_names))
    indices = np.argsort(importance)[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(top_n), importance[indices])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Top-{top_n} Feature Importance")
    path = str(Path(reports_dir) / "feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_evaluation(
    estimator,
    X_test,
    y_test,
    y_pred,
    y_prob,
    reports_dir: str = REPORTS_DIR,
) -> dict:
    return {
        "confusion_matrix": plot_confusion_matrix(y_test, y_pred, reports_dir),
        "roc_curve": plot_roc_curve(y_test, y_prob, reports_dir),
        "feature_importance": plot_feature_importance(estimator, list(X_test.columns), reports_dir),
    }
