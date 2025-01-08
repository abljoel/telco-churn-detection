"""Linear Classifier models for churn detection."""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LinearModel(BaseModel):
    """Linear Model utilizing Logistic Regression.

    Attributes:
        model: Instance of the scikit-learn LogisticRegression model.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LinearModel with given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super().__init__(config)
        self.model = LogisticRegression(**self.config.get("model_params", {}))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the provided dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes using the trained model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input data using the model.

        This method computes class probabilities for each input sample using the `predict_proba`
        method of the underlying model. The model must implement the `predict_proba` method.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class probabilities
        """
        return self.model.predict_proba(X)


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models to integrate with BaseModel interface.

    Attributes:
        model_class: The scikit-learn model class to be instantiated.
        model: The instantiated scikit-learn model.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **params: Union[str, int, float, Dict[str, Any]],
    ):
        """Initialize the SklearnModel with configuration or parameters.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary for the model.
            **params: Additional parameters for the model when config is not provided.
        """
        if config is not None:
            super().__init__(config)
            self.model_class = self._get_model_class()
            model_params = self.config.get("params", {})
        else:
            self.model_class = LogisticRegression
            model_params = params

        self.model = self.model_class(**model_params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model with the provided dataset.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        self.fit(X, y)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        """Fit the model to the dataset.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            SklearnModel: The instance of the trained model.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outcomes using the trained model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input data, if supported by the model.

        Args:
            X (pd.DataFrame): Input data as a pandas DataFrame, where rows represent samples
                            and columns represent features.

        Returns:
            np.ndarray: A 2D NumPy array where each row contains the predicted class probabilities
                        for the corresponding input sample. The array shape is (n_samples,
                        n_classes).

        Raises:
            NotImplementedError: If the underlying model does not implement `predict_proba` and the
                                base class's `predict_proba` method is called.
        Notes:
            - If the model has a `predict_proba` method, it will be used directly.
            - If not, this method defers to the base class implementation, which is expected to
              raise an error unless explicitly overridden.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return super().predict_proba(X)  # Will raise NotImplementedError

    def set_params(self, **params: Any) -> "SklearnModel":
        """Set parameters for the underlying model.

        Args:
            **params (Any): Parameters to set on the model.

        Returns:
            SklearnModel: The instance of the model with updated parameters.
        """
        self.model.set_params(**params)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters of the underlying model.

        Args:
            deep (bool): Whether to include parameters of sub-objects.

        Returns:
            Dict[str, Any]: Parameters of the model.
        """
        return self.model.get_params(deep=deep)

    def _get_model_class(self) -> Any:
        """Retrieve the appropriate model class based on configuration.

        Returns:
            Any: The scikit-learn model class.

        Raises:
            ValueError: If an unknown model type is specified in the config.
        """
        model_type = self.config.get("type")
        if model_type == "logistic_regression":
            return LogisticRegression
        raise ValueError(f"Unknown model type: {model_type}")
