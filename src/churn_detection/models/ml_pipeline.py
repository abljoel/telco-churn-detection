"""ML Pipeline for churn detection."""

from typing import List, Dict, Callable, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from .base_model import BaseModel, BaseFeatureEngineer, ColumnPreprocessorFeatures
from .linear_classifier import SklearnModel


METRIC_MAP = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
}


class MLPipeline:
    """
    A machine learning pipeline for handling feature engineering, model training,
    prediction, and evaluation.

    Attributes:
        feature_engineers (List[BaseFeatureEngineer]): List of feature engineering steps.
        model (BaseModel): The model used for training and prediction.
        metrics (Dict[str, Callable]): Dictionary mapping metric names to metric functions.
    """

    def __init__(
        self,
        feature_engineers: List[BaseFeatureEngineer],
        model: BaseModel,
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    ) -> None:
        """
        Initializes the MLPipeline.

        Args:
            feature_engineers (List[BaseFeatureEngineer]): List of feature engineering steps.
            model (BaseModel): The model for training and prediction.
            metrics (Dict[str, Callable]): Metric functions for evaluation.
        """
        self.feature_engineers = feature_engineers
        self.model = model
        self.metrics = metrics

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the pipeline by applying feature engineering and training the model.

        Args:
            X (np.ndarray): Feature data for training.
            y (np.ndarray): Target labels for training.
        """
        X_transformed = X
        for engineer in self.feature_engineers:
            X_transformed = engineer.fit_transform(X_transformed)
        self.model.train(X_transformed, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained pipeline.

        Args:
            X (np.ndarray): Feature data for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        X_transformed = X
        for engineer in self.feature_engineers:
            X_transformed = engineer.transform(X_transformed)
        return self.model.predict(X_transformed)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the pipeline using the provided metrics.

        Args:
            X (np.ndarray): Feature data for evaluation.
            y_true (np.ndarray): True target labels.

        Returns:
            Dict[str, float]: Evaluation results for each metric.
        """
        y_pred = self.predict(X)
        results = {
            metric_name: metric_func(y_true, y_pred)
            for metric_name, metric_func in self.metrics.items()
        }
        return results


def create_pipeline(config: Dict[str, Any]) -> MLPipeline:
    """
    Creates and configures a machine learning pipeline based on a provided configuration.

    This function reads the configuration dictionary to initialize feature engineering steps,
    the machine learning model, and evaluation metrics for the pipeline. The pipeline is built
    with modular components that can be dynamically adapted based on the configuration settings.

    Args:
        config (Dict[str, Any]): A dictionary containing the pipeline configuration. It should include:
            - `feature_engineering`: A dictionary with:
                - `type` (str): The type of feature engineering (e.g., "column_preprocessor").
                - `params` (Dict): Parameters for feature engineering, including variable types and steps.
            - `model`: A dictionary with:
                - `type` (str): The type of model (e.g., "logistic_regression").
                - `params` (Dict): Model-specific parameters.
            - `metrics` (List[str]): A list of metric labels to evaluate the model. Supported labels include:
                - "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc".

    Returns:
        MLPipeline: A configured machine learning pipeline object ready for training and evaluation.

    Raises:
        ValueError: If the feature engineering type is unsupported.
        ValueError: If no valid metrics are found in the configuration.
    """
    feature_engineering_config = config.get("feature_engineering", {})
    feature_engineer_type = feature_engineering_config.get("type")

    if feature_engineer_type == "column_preprocessor":
        feature_engineer = ColumnPreprocessorFeatures(
            feature_engineering_config.get("params", {})
        )
    else:
        raise ValueError(f"Unknown feature engineering type: {feature_engineer_type}")

    model = SklearnModel(config.get("model"))

    metric_labels = config.get("metrics", [])
    metrics = {
        label: METRIC_MAP[label] for label in metric_labels if label in METRIC_MAP
    }

    if not metrics:
        raise ValueError("No valid metrics found in configuration.")

    return MLPipeline(
        feature_engineers=[feature_engineer], model=model, metrics=metrics
    )
