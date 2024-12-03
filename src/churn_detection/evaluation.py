"""
Python module for model evaluation utilities.
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, classification_report


def display_roc_auc_score(
    model: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
) -> None:
    """
    Display the ROC AUC score for a given model and dataset.

    Args:
        model (BaseEstimator): A scikit-learn-like model that implements a predict method.
        X (Union[np.ndarray, pd.DataFrame]): Features as a NumPy array or pandas DataFrame.
        y (Union[np.ndarray, pd.Series]): True labels as a NumPy array or pandas Series.

    Raises:
        ValueError: If the model does not have a `predict` method.
        ValueError: If `X` or `y` are empty.
    """
    if not hasattr(model, "predict"):
        raise ValueError("The provided model does not have a predict method.")

    if X is None or len(X) == 0:
        raise ValueError("The input feature data 'X' cannot be empty.")

    if y is None or len(y) == 0:
        raise ValueError("The target labels 'y' cannot be empty.")

    y_pred = model.predict(X)
    roc_auc = roc_auc_score(y, y_pred)
    print(f"ROC AUC score: {roc_auc:.2f}")


def display_clf_report(
    y_pred: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
) -> None:
    """
    Display the classification report for given predictions and true labels.

    Args:
        y_pred (Union[np.ndarray, pd.Series]): Predicted labels as a NumPy array or pandas Series.
        y (Union[np.ndarray, pd.Series]): True labels as a NumPy array or pandas Series.

    Raises:
        ValueError: If `y_pred` or `y` are empty.
    """
    if y_pred is None or len(y_pred) == 0:
        raise ValueError("The predicted labels 'y_pred' cannot be empty.")

    if y is None or len(y) == 0:
        raise ValueError("The target labels 'y' cannot be empty.")

    print(classification_report(y, y_pred))
