"""ML Pipeline for churn detection."""

from typing import List, Dict, Callable, Any
import numpy as np
from sklearn.metrics import f1_score, recall_score
from .base_model import BaseModel, BaseFeatureEngineer, ColumnPreprocessorFeatures
from .linear_classifier import SklearnModel


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
    Create an MLPipeline instance from a configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing feature engineering,
        model, and metric specifications.

    Returns:
        MLPipeline: Configured machine learning pipeline.

    Raises:
        ValueError: If an unknown feature engineering type is specified in the config.
    """
    # Create feature engineers
    feature_engineering_config = config.get("feature_engineering", {})
    feature_engineer_type = feature_engineering_config.get("type")

    if feature_engineer_type == "column_preprocessor":
        feature_engineer = ColumnPreprocessorFeatures(
            feature_engineering_config.get("params", {})
        )
    else:
        raise ValueError(f"Unknown feature engineering type: {feature_engineer_type}")

    # Create model
    model = SklearnModel(config.get("model"))

    # Define metrics
    metrics = {
        metric_name: metric_func
        for metric_name, metric_func in zip(
            config.get("metrics", ["f1", "recall"]),
            [
                f1_score,
                lambda y_true, y_pred: recall_score(
                    y_true, y_pred, pos_label=1, average="binary"
                ),
            ],
        )
    }

    return MLPipeline(
        feature_engineers=[feature_engineer], model=model, metrics=metrics
    )
