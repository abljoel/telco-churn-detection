"""Base Model Utilities"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import numpy as np
from sklearn.pipeline import Pipeline
from churn_detection.features import Transformation, ColumnPreprocessor


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: Dict):
        """
        Initialize the BaseModel with a configuration.

        Args:
            config (Dict): Configuration dictionary for the model.
        """
        self.config = config
        self.model: Optional[object] = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions.
        Not all models support this, so it's not an abstract method.
        """
        raise NotImplementedError("This model doesn't support probability predictions")


class BaseFeatureEngineer(ABC):
    """Abstract base class for feature engineering."""

    def __init__(self, config: Dict):
        """
        Initialize the BaseFeatureEngineer with a configuration.

        Args:
            config (Dict): Configuration dictionary for feature engineering.
        """
        self.config = config

    @abstractmethod
    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit and transform features.

        Args:
            X (np.ndarray): Input features.
            y (Optional[np.ndarray]): Target values (optional).

        Returns:
            np.ndarray: Transformed features.
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Transformed features.
        """
        pass


class ColumnPreprocessorFeatures(BaseFeatureEngineer):
    """
    Wrapper for the column preprocessor system that implements the BaseFeatureEngineer interface.
    """

    def __init__(self, config: Optional[Dict] = None, **kwargs):
        """
        Initialize the ColumnPreprocessorFeatures with a configuration.

        Args:
            config (Optional[Dict]): Configuration dictionary for the column preprocessor.
            **kwargs: Additional keyword arguments passed during parameter tuning.
        """
        config = config or {}

        combined_config = {**config, **kwargs}
        super().__init__(combined_config)

        self.preprocessor: Optional[Pipeline] = None
        self._setup_preprocessor()

    def _setup_preprocessor(self) -> None:
        """Setup the column preprocessor based on the provided configuration."""
        preprocessor = ColumnPreprocessor(
            remainder=self.config.get("remainder", "drop"),
            sparse_threshold=self.config.get("sparse_threshold", 0.3),
            n_jobs=self.config.get("n_jobs", None),
            transformer_weights=self.config.get("transformer_weights", None),
            verbose_feature_names_out=self.config.get(
                "verbose_feature_names_out", True
            ),
            force_int_remainder_cols=self.config.get("force_int_remainder_cols", True),
        )

        if "numerical" in self.config:
            try:
                preprocessor.add_transformation(
                    Transformation(**self.config["numerical"])
                )
            except Exception as e:
                raise ValueError(f"Error adding numerical transformations: {e}")

        if "categorical" in self.config:
            try:
                preprocessor.add_transformation(
                    Transformation(**self.config["categorical"])
                )
            except Exception as e:
                raise ValueError(f"Error adding categorical transformations: {e}")

        self.preprocessor = preprocessor.create_preprocessor()

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.

        Args:
            X (np.ndarray): Input features.
            y (Optional[np.ndarray]): Target values (optional).

        Returns:
            np.ndarray: Transformed features.
        """
        if self.preprocessor is None:
            self._setup_preprocessor()
        return self.preprocessor.fit_transform(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Transformed features.
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor not initialized. Call fit() or fit_transform() first."
            )
        return self.preprocessor.transform(X)

    def __reduce__(self):
        """Make the class picklable."""
        return (self.__class__, (self.config,))

    def __getstate__(self):
        """Return state for pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Set state during unpickling."""
        self.__dict__.update(state)

    def set_params(self, **params: Any) -> "ColumnPreprocessorFeatures":
        """
        Set the parameters of this preprocessor.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        # Split parameters into different categories
        nested_params = {k: v for k, v in params.items() if "__" in k}
        config_params = {k: v for k, v in params.items() if "__" not in k}

        if config_params:
            self.config.update(config_params)
            self._setup_preprocessor()

        # Update nested parameters in the preprocessor
        if nested_params and self.preprocessor is not None:
            self.preprocessor.set_params(**nested_params)

        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this preprocessor.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                        contained sub-objects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        params = self.config.copy()
        if deep and self.preprocessor is not None:
            preprocessor_params = self.preprocessor.get_params(deep=deep)
            params.update(preprocessor_params)

        return params

    @staticmethod
    def _get_feature_names(
        column_transformer: Pipeline, feature_names: List[str]
    ) -> List[str]:
        """
        Extract feature names from the column transformer.

        Args:
            column_transformer (Pipeline): Fitted column transformer.
            feature_names (List[str]): Original feature names.

        Returns:
            List[str]: Output feature names.
        """
        output_features = []

        for name, pipe, features in column_transformer.transformers_:
            if name != "remainder":
                current_features = features
                if isinstance(pipe, Pipeline):
                    for step in pipe.steps:
                        if hasattr(step[1], "get_feature_names_out"):
                            current_features = step[1].get_feature_names_out(
                                current_features
                            )
                    output_features.extend(current_features)
                elif hasattr(pipe, "get_feature_names_out"):
                    output_features.extend(pipe.get_feature_names_out(features))
                else:
                    output_features.extend(features)

        return output_features
