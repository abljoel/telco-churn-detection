"""Feature Pipeline Utilities"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    A dummy transformer that performs no transformation.
    This can be used as a placeholder or for testing purposes.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
    ) -> "DummyTransformer":
        """
        Fit the transformer.

        Parameters:
        X: Union[pd.DataFrame, np.ndarray], optional
            Input features (DataFrame or ndarray).
        y: Union[pd.Series, np.ndarray], optional
            Target labels (ignored).

        Returns:
        self
        """
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the input data (no changes are made).

        Parameters:
        X: Union[pd.DataFrame, np.ndarray]
            Input features to be transformed.

        Returns:
        X: Union[pd.DataFrame, np.ndarray]
            The input data unchanged.

        Raises:
        TypeError: If input is not a pandas DataFrame or numpy ndarray.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input should be a pandas DataFrame or a numpy ndarray")
        return X


class DummyNumericTransformer(DummyTransformer):
    """
    A dummy transformer that performs no transformation but checks if the input data is numeric.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
    ) -> "DummyNumericTransformer":
        """
        Fit the transformer and validate if the input data is numeric.

        Parameters:
        X: Union[pd.DataFrame, np.ndarray], optional
            Input features (DataFrame or ndarray).
        y: Union[pd.Series, np.ndarray], optional
            Target labels (ignored).

        Returns:
        self

        Raises:
        ValueError: If the input data is not numeric.
        """
        if isinstance(X, pd.DataFrame):
            if not all([pd.api.types.is_numeric_dtype(X[col]) for col in X.columns]):
                raise ValueError(
                    "DummyNumericTransformer only supports numeric columns."
                )
        elif isinstance(X, np.ndarray):
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("DummyNumericTransformer only supports numeric data.")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the input data by selecting only numeric columns (if input is a DataFrame).

        Parameters:
        X: Union[pd.DataFrame, np.ndarray]
            Input features to be transformed.

        Returns:
        X: Union[pd.DataFrame, np.ndarray]
            Numeric columns of the input data.
        """
        X = super().transform(X)
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number])
        elif isinstance(X, np.ndarray):
            return X
        return X


class SimpleCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    A simple transformer that encodes categorical features using sklearn's LabelEncoder.
    """

    def __init__(self) -> None:
        self.encoders = {}

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ) -> "SimpleCategoryEncoder":
        """
        Fit the LabelEncoder for each column in the input data.

        Parameters:
        X: Union[pd.DataFrame, np.ndarray]
            Input features to be encoded.
        y: Union[pd.Series, np.ndarray], optional
            Target labels (ignored).

        Returns:
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.copy()

        for col in range(X.shape[1]):
            le = LabelEncoder()
            le.fit(X.iloc[:, col] if isinstance(X, pd.DataFrame) else X[:, col])
            self.encoders[col] = le
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the input data by encoding categorical columns.

        Parameters:
        X: Union[pd.DataFrame, np.ndarray]
            Input features to be transformed.

        Returns:
        X_out: Union[pd.DataFrame, np.ndarray]
            Encoded categorical data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.copy()

        X_out = X.copy()
        for col in range(X_out.shape[1]):
            X_out.iloc[:, col] = self.encoders[col].transform(
                X_out.iloc[:, col] if isinstance(X_out, pd.DataFrame) else X_out[:, col]
            )
        return X_out


def add_transformation(
    name: str,
    steps: List[Tuple[str, TransformerMixin]],
    variables: List[str],
) -> Tuple[str, Pipeline, List[str]]:
    """
    Create a named transformer pipeline.

    Parameters:
    name: str
        The name of the transformer.
    steps: List[Tuple[str, TransformerMixin]]
        A list of steps for the transformer pipeline.
    variables: List[str]
        A list of target variables to be included in the pipeline.

    Returns:
    Tuple[str, Pipeline, List[str]]
        A tuple containing the name, the transformer pipeline, and the list of target variables.
    """
    transformer = Pipeline(steps=steps)
    return name, transformer, variables


def create_column_preprocessor(
    transformers: List[Tuple[str, Pipeline, List[str]]]
) -> ColumnTransformer:
    """
    Create a column transformer preprocessor.

    Parameters:
    transformers: List[Tuple[str, Pipeline, List[str]]]
        A list of transformers, each defined as a tuple of (name, transformer, columns).

    Returns:
    ColumnTransformer
        A column transformer for preprocessing different columns.
    """
    return ColumnTransformer(transformers=transformers)


def create_pipe(prep: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    """
    Create a pipeline that includes a preprocessor and a model.

    Parameters:
    prep: ColumnTransformer
        The preprocessor used to transform the input data.
    model: BaseEstimator
        The model to be used as the final estimator in the pipeline.

    Returns:
    Pipeline
        A scikit-learn pipeline consisting of the preprocessor and the model.
    """
    model = Pipeline(steps=[("processor", prep), ("estimator", model)])
    return model
