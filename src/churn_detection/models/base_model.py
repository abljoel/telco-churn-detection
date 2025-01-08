"""Base Model Utilities"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from ..features import Transformation, ColumnPreprocessor


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
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform features.

        Args:
            X (np.ndarray): Input features.

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

    def __init__(self, config: Dict):
        """
        Initialize the ColumnPreprocessorFeatures with a configuration.

        Args:
            config (Dict): Configuration dictionary for the column preprocessor.
        """
        super().__init__(config)
        self.preprocessor: Optional[Pipeline] = None
        self._setup_preprocessor()

    def _setup_preprocessor(self) -> None:
        """
        Setup the column preprocessor based on the provided configuration.
        """
        preprocessor = ColumnPreprocessor()

        if "numerical" in self.config:
            preprocessor.add_transformation(Transformation(**self.config["numerical"]))

        if "categorical" in self.config:
            preprocessor.add_transformation(
                Transformation(**self.config["categorical"])
            )

        self.preprocessor = preprocessor.create_preprocessor()

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ColumnPreprocessorFeatures":
        """
        Fit the preprocessor to the data.

        Args:
            X (pd.DataFrame): Input features.
            y (Optional[pd.Series]): Target values (optional).

        Returns:
            ColumnPreprocessorFeatures: The fitted preprocessor.
        """
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Transformed features as a DataFrame.
        """
        transformed_data = self.preprocessor.transform(X)
        return pd.DataFrame(
            transformed_data,
            columns=self._get_feature_names(self.preprocessor, X.columns),
        )

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.

        Args:
            X (pd.DataFrame): Input features.
            y (Optional[pd.Series]): Target values (optional).

        Returns:
            pd.DataFrame: Transformed features as a DataFrame.
        """
        transformed_data = self.preprocessor.fit_transform(X)
        return pd.DataFrame(
            transformed_data,
            columns=self._get_feature_names(self.preprocessor, X.columns),
        )

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
