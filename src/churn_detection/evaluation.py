"""
Python module for model evaluation utilities.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .preprocessing import split_data


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


def validate_model_with_cv(
    model: BaseEstimator,
    train_data: Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame],
    n_folds: int = 10,
    metric: str = "accuracy",
    results: bool = False,
) -> Tuple[float, float]:
    """Validates a model using repeated stratified K-fold cross-validation.

    This function performs cross-validation on a given model using a repeated stratified
    K-fold approach. It supports both scenarios where the data is already split into
    features (X) and target (y), or where a splitting function is required to extract
    features and target from a full dataset.

    Args:
        model (BaseEstimator): The machine learning model or pipeline to be validated.
            This should be an estimator compatible with scikit-learn's `fit` and `predict` methods.
        train_data (Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]):
            Training data which can either be:
            - A tuple containing the feature matrix `X_train` and target vector `y_train`.
            - A full dataset `pd.DataFrame` requiring splitting into features and target.
        n_folds (int, optional): The number of splits for K-fold cross-validation. Defaults to 10.
        metric (str, optional): Scoring metric to evaluate model performance. This should be a valid
                                scoring parameter for scikit-learn's `cross_val_score`.
                                Defaults to "accuracy".
        results (bool, optional): If `True`, returns the list of cross-validation scores.
                                  Defaults to `False`.

    Returns:
        Union[Tuple[float, float], List[float]]: Depending on the value of the `results` parameter:
            - If `results` is `False`: A tuple containing:
                - `mean_score` (float): The mean of the cross-validation scores.
                - `std_dev` (float): The standard deviation of the cross-validation scores.
            - If `results` is `True`: A list of cross-validation scores.

    Raises:
        TypeError: If `train_data` is not correctly formatted or if the split function
                   returns invalid data types.
    """

    # Split the training data if required
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
    else:
        X_train, y_train = split_data(train_data)

    # Ensure the split function returns valid data types
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)) or not isinstance(
        y_train, (pd.Series, np.ndarray)
    ):
        raise TypeError(
            "The split function must return a DataFrame/ndarray for X and Series/ndarray for y."
        )

    # Set up the cross-validation strategy
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)

    # Calculate cross-validation scores
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=metric)
    if results:
        return cv_results

    # Output results
    mean_score = cv_results.mean()
    std_dev = cv_results.std()

    return mean_score, std_dev
