import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils import plot_confusion_matrix  # Import from utils

# The evaluate_model, predict_prob, predict functions are now in train.py
# because they are tightly coupled with the training pipeline steps in the original notebook.
# Keeping them here is also a valid structure if you prefer evaluate.py for ALL eval tasks.
# For now, let's follow the flow where train.py calls these after training phases.
# You could move them back here if needed and import them into train.py.


def calculate_metrics(y_true, y_pred, average="weighted"):
    """
    Calculates and prints classification metrics.

    Args:
        y_true (np.ndarray): True labels (class indices).
        y_pred (np.ndarray): Predicted labels (class indices).
        average (str): Type of averaging for metrics ('weighted', 'macro', 'micro', etc.).
    """
    print("Calculating metrics:")
    try:
        precision = precision_score(y_true, y_pred, average=average)
        print(f"Precision ({average}): {precision:.4f}")

        recall = recall_score(y_true, y_pred, average=average)
        print(f"Recall ({average}):    {recall:.4f}")

        f1 = f1_score(y_true, y_pred, average=average)
        print(f"F1 Score ({average}):  {f1:.4f}")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print(
            "Cannot calculate weighted average metrics if a class has no true/predicted samples."
        )
        print("Consider using average='macro' or average=None.")


def print_classification_report(y_true, y_pred, classes):
    """
    Prints the classification report.

    Args:
        y_true (np.ndarray): True labels (class indices).
        y_pred (np.ndarray): Predicted labels (class indices).
        classes (list): List of class names.
    """
    print("\nClassification Report:")
    # Ensure target_names matches the order of class indices
    print(classification_report(y_true, y_pred, target_names=classes))


if __name__ == "__main__":
    # Example Usage:
    # Create dummy true and predicted labels
    from config import cfg

    y_true_dummy = np.random.randint(0, cfg.NUM_CLASSES, size=100)
    y_pred_dummy = np.random.randint(0, cfg.NUM_CLASSES, size=100)

    calculate_metrics(y_true_dummy, y_pred_dummy, average="weighted")
    print_classification_report(y_true_dummy, y_pred_dummy, cfg.CLASSES)

    # Create a dummy confusion matrix
    cm_dummy = confusion_matrix(y_true_dummy, y_pred_dummy)
    plot_confusion_matrix(cm_dummy, cfg.CLASSES, title="Dummy Confusion Matrix")
