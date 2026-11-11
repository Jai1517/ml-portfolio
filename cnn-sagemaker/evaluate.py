"""Model evaluation: accuracy, per-class report, confusion matrix."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(y_true, y_pred, labels=None):
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, target_names=labels))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return accuracy_score(y_true, y_pred)
