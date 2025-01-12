"""ML Pipeline for churn detection."""

from typing import List, Dict, Callable, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from churn_detection.models.base_model import (
    BaseModel,
    BaseFeatureEngineer,
    ColumnPreprocessorFeatures,
)
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
    A machine learning pipeline for feature engineering, model training,
    prediction, and evaluation.

    Attributes:
        feature_engineers (List[BaseFeatureEngineer]):
            List of feature engineering steps to apply to the data.
        model (BaseModel):
            Machine learning model to use for training and prediction.
        metrics (Dict[str, Callable[[np.ndarray, np.ndarray], float]]):
            Dictionary of metric names and corresponding evaluation functions.
        _pipeline (Pipeline):
            Internal scikit-learn pipeline combining feature engineering and the model.
    """

    def __init__(
        self,
        feature_engineers: List["BaseFeatureEngineer"],
        model: "BaseModel",
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    ) -> None:
        """
        Initialize the MLPipeline with feature engineering steps, a model, and evaluation metrics.

        Args:
            feature_engineers (List[BaseFeatureEngineer]):
                List of feature engineering steps to apply to the data.
            model (BaseModel):
                Machine learning model to use for training and prediction.
            metrics (Dict[str, Callable[[np.ndarray, np.ndarray], float]]):
                Dictionary of metric names and their corresponding functions.
        """
        self.feature_engineers = feature_engineers
        self.model = model
        self.metrics = metrics
        self._pipeline = None
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """
        Set up the scikit-learn pipeline with feature engineering steps and the model.
        """
        steps = [
            (f"feature_engineer", engineer)
            for i, engineer in enumerate(self.feature_engineers)
        ]
        steps.append(("model", self.model))
        self._pipeline = Pipeline(steps=steps)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the pipeline by applying feature engineering and fitting the model.

        Args:
            X (np.ndarray):
                Input features.
            y (np.ndarray):
                Target values.
        """
        self._pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained pipeline.

        Args:
            X (np.ndarray):
                Input features.

        Returns:
            np.ndarray: Predictions made by the model.
        """
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the trained pipeline.

        Args:
            X (np.ndarray):
                Input features.

        Returns:
            np.ndarray: Predicted probabilities for each class.

        Raises:
            AttributeError: If the underlying model does not support probability predictions.
        """
        if not hasattr(self._pipeline, "predict_proba"):
            raise AttributeError(
                "The underlying model does not support probability predictions."
            )
        return self._pipeline.predict_proba(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the pipeline using the provided metrics.

        Args:
            X (np.ndarray):
                Input features.
            y_true (np.ndarray):
                True target values.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation results for each metric.
        """
        y_pred = self.predict(X)
        return {
            metric_name: metric_func(y_true, y_pred)
            for metric_name, metric_func in self.metrics.items()
        }

    def get_pipeline(self) -> Pipeline:
        """
        Retrieve the underlying scikit-learn pipeline.

        Returns:
            Pipeline: The internal scikit-learn pipeline.
        """
        return self._pipeline


def create_pipeline(config: Dict[str, Any]) -> MLPipeline:
    """
    Creates and configures a machine learning pipeline based on a provided configuration.

    This function reads the configuration dictionary to initialize feature engineering steps,
    the machine learning model, and evaluation metrics for the pipeline. The pipeline is built
    with modular components that can be dynamically adapted based on the configuration settings.

    Args:
        config (Dict[str, Any]): A dictionary containing the pipeline configuration. It should
            include:
            - `feature_engineering`: A dictionary with:
                - `type` (str): The type of feature engineering (e.g., "column_preprocessor").
                - `params` (Dict): Parameters for feature engineering, including variable types and
                   steps.
            - `model`: A dictionary with:
                - `type` (str): The type of model (e.g., "logistic_regression").
                - `params` (Dict): Model-specific parameters.
            - `metrics` (List[str]): A list of metric labels to evaluate the model. Supported labels
               include:
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
