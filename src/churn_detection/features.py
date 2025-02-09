"""Feature Pipeline Utilities"""

from typing import List, Tuple, Union, Optional
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


# class FeatureRemover(BaseEstimator, TransformerMixin):
#     """
#     A transformer that removes specified features from the dataset.
#     """

#     def __init__(self, features_to_remove: Union[str, List[str]]) -> None:
#         """
#         Initialize the transformer with the features to remove.

#         Parameters:
#         features_to_remove: Union[str, List[str]]
#             The name or list of names of the features to remove from the dataset.
#         """
#         if isinstance(features_to_remove, str):
#             self.features_to_remove = [features_to_remove]
#         elif isinstance(features_to_remove, list):
#             self.features_to_remove = features_to_remove
#         else:
#             raise ValueError("features_to_remove must be a string or a list of strings")

#     def fit(
#         self,
#         X: Union[pd.DataFrame, np.ndarray] = None,
#         y: Union[pd.Series, np.ndarray] = None,
#     ) -> "FeatureRemover":
#         """
#         Fit the transformer. This method does not learn anything as it's used for column removal.

#         Parameters:
#         X: Union[pd.DataFrame, np.ndarray], optional
#             Input features (DataFrame or ndarray).
#         y: Union[pd.Series, np.ndarray], optional
#             Target labels (ignored).

#         Returns:
#         self
#         """
#         # No fitting required as we are only removing features
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Transform the input data by removing the specified features.

#         Parameters:
#         X: pd.DataFrame
#             Input features to be transformed.

#         Returns:
#         X_transformed: pd.DataFrame
#             The input data with the specified features removed.

#         Raises:
#         TypeError: If input is not a pandas DataFrame.
#         KeyError: If any feature to remove is not in the DataFrame.
#         """
#         if not isinstance(X, pd.DataFrame):
#             raise TypeError("Input should be a pandas DataFrame")

#         missing_features = [
#             feature for feature in self.features_to_remove if feature not in X.columns
#         ]
#         if missing_features:
#             raise KeyError(f"Features {missing_features} not found in the dataset")

#         return X.drop(columns=self.features_to_remove)


class InteractionStrengthExtractor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to compute and optionally encode interaction strength
    between two categorical variables based on the mean value of a target variable.

    This transformer allows you to:
    - Compute interaction strength for pairs of categorical variables.
    - Optionally apply ordinal encoding to the computed interaction strength values.
    - Integrate the computed interaction strength into the input DataFrame.

    Attributes:
        cat_column_1 (str): Name of the first categorical column.
        cat_column_2 (str): Name of the second categorical column.
        target_column (str): Name of the target column used for computing interaction strength.
        ordinal_encode (bool): Whether to apply ordinal encoding to interaction strength values.
        interaction_strength_ (pd.DataFrame): DataFrame containing the computed interaction strength
                                              values.
        label_encoder_ (LabelEncoder or None): LabelEncoder instance if ordinal encoding is applied.
        strength_col_name_ (str): Name of the generated interaction strength column.
    """

    def __init__(
        self,
        cat_column_1: str,
        cat_column_2: str,
        target_column: str = "churn",
        ordinal_encode: bool = False,
    ):
        """
        Initializes the InteractionStrengthExtractor.

        Args:
            cat_column_1 (str): Name of the first categorical column.
            cat_column_2 (str): Name of the second categorical column.
            target_column (str): Name of the target column (e.g., churn).
            ordinal_encode (bool): Whether to apply ordinal encoding to interaction strength values.
        """
        self.cat_column_1 = cat_column_1
        self.cat_column_2 = cat_column_2
        self.target_column = target_column
        self.ordinal_encode = ordinal_encode
        self.interaction_strength_ = None
        self.label_encoder_ = LabelEncoder() if ordinal_encode else None
        self.strength_col_name_ = None

    def fit(self, X, y=None):
        """
        Compute interaction strength as the mean target value for each combination
        of the two categorical columns.

        Args:
            X (pd.DataFrame): Input dataframe containing the categorical columns and target column.
            y (pd.Series, optional): Not used (included for compatibility).

        Returns:
            self
        """
        if not all(
            col in X.columns
            for col in [self.cat_column_1, self.cat_column_2, self.target_column]
        ):
            raise ValueError(
                f"Columns {self.cat_column_1}, {self.cat_column_2}, and {self.target_column} "
                f"must be in the input DataFrame."
            )

        self.strength_col_name_ = f"{self.cat_column_1}_{self.cat_column_2}_strength"
        self.interaction_strength_ = (
            X.groupby([self.cat_column_1, self.cat_column_2])[self.target_column]
            .mean()
            .reset_index()
        )
        self.interaction_strength_.rename(
            columns={self.target_column: self.strength_col_name_}, inplace=True
        )

        if self.ordinal_encode:
            self.interaction_strength_[self.strength_col_name_] = (
                self.label_encoder_.fit_transform(
                    self.interaction_strength_[self.strength_col_name_]
                )
            )

        return self

    def transform(self, X):
        """
        Merge the interaction strength information into the input DataFrame.

        Args:
            X (pd.DataFrame): Input dataframe containing the categorical columns.

        Returns:
            pd.DataFrame: Transformed dataframe with an additional interaction strength column.
        """
        if self.interaction_strength_ is None:
            raise RuntimeError(
                "The transformer has not been fitted yet. Call 'fit' before 'transform'."
            )

        X_transformed = X.merge(
            self.interaction_strength_,
            on=[self.cat_column_1, self.cat_column_2],
            how="left",
        )
        return X_transformed

    def get_strength_col_name(self):
        """
        Get the name of the interaction strength column.

        Returns:
            str: Name of the interaction strength column.
        """
        if self.strength_col_name_ is None:
            raise RuntimeError(
                "The transformer has not been fitted yet. Call 'fit' before accessing the "
                "strength column name."
            )
        return self.strength_col_name_


class FeatureConcatenator(BaseEstimator, TransformerMixin):
    def __init__(self, feature_pairs: List[Tuple[str, str]]) -> None:
        """
        Custom Transformer for creating interaction features by concatenating pairs of categorical
        variables.

        Parameters:
        - feature_pairs: List of tuples, where each tuple contains two feature names to concatenate.
          Example: [("feature1", "feature2"), ("feature3", "feature4")]
        """
        self.feature_pairs = feature_pairs
        self.new_feature_names = []

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FeatureConcatenator":
        """
        Fit method, does nothing as this transformer doesn't require fitting.

        Parameters:
        - X: Input DataFrame.
        - y: Optional target variable, not used.

        Returns:
        - self: Returns the instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataset by adding concatenated interaction features.

        Parameters:
        - X: Input DataFrame

        Returns:
        - Transformed DataFrame with new concatenated features added.
        """
        X = X.copy()
        self.new_feature_names = []  # Reset feature names
        for feature1, feature2 in self.feature_pairs:
            new_feature_name = f"{feature1}_{feature2}_concat"
            self.new_feature_names.append(new_feature_name)
            X[new_feature_name] = (
                X[feature1].astype(str) + "_" + X[feature2].astype(str)
            )
        return X

    def get_new_feature_names(self) -> List[str]:
        """
        Get the names of the newly created concatenated features.

        Returns:
        - List of new feature names.
        """
        return self.new_feature_names


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol: float = 0.02, replace_with: str = "Rare") -> None:
        """
        Transformer to encode rare categories in categorical features by replacing
        them with a specified value.

        Args:
            tol (float): The minimum frequency (proportion) for a category to be considered
                         frequent.
            replace_with (str): The value to replace rare categories with.
        """
        self.tol = tol
        self.replace_with = replace_with

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None
    ) -> "RareCategoryEncoder":
        """
        Fit method does nothing as this transformer doesn't require fitting.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.
            y (Optional[pd.Series]): Target variable, not used.

        Returns:
            RareCategoryEncoder: The fitted instance of the transformer.
        """
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the dataset by encoding rare categories.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Transformed data with rare categories replaced.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input should be a pandas DataFrame or a numpy ndarray.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feat{i}" for i in range(X.shape[1])])

        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")

        X = X.copy()

        for feature in X.columns:
            freqs = X[feature].value_counts(normalize=True)
            frequent_categories = freqs[freqs >= self.tol].index

            X[feature] = np.where(
                X[feature].isin(frequent_categories), X[feature], self.replace_with
            )

        return X.values if isinstance(X, pd.DataFrame) else X


class Transformation:
    """
    A class to represent a named transformer pipeline, which encapsulates a series of transformation
    steps and the target variables.
    """

    def __init__(
        self, name: str, steps: List[Tuple[str, TransformerMixin]], variables: List[str]
    ):
        """
        Create a named transformer pipeline.

        Parameters:
        name: str
            The name of the transformer.
        steps: List[Tuple[str, TransformerMixin]]
            A list of steps for the transformer pipeline.
        variables: List[str]
            A list of target variables to be included in the pipeline.
        """
        self.name = name
        self.steps = steps
        self.variables = variables
        self.pipeline = Pipeline(steps=steps)

    def get_tuple(self) -> Tuple[str, Pipeline, List[str]]:
        """
        Get the tuple representation of the transformation.

        Returns:
        Tuple[str, Pipeline, List[str]]
            A tuple containing the name, the transformer pipeline, and the list of target variables.
        """
        return self.name, self.pipeline, self.variables

    def __reduce__(self):
        return (self.__class__, (self.name, self.steps, self.variables))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ColumnPreprocessor:
    """
    A class to manage and create a column transformer preprocessor.
    """

    def __init__(
        self,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose_feature_names_out=True,
        force_int_remainder_cols=True,
    ):
        """
        Initialize an empty list of transformations.

        Args:
            remainder: {'drop', 'passthrough'} or estimator, default='drop'
            sparse_threshold: float, default=0.3
            n_jobs: int, default=None
            transformer_weights: dict, default=None
            verbose_feature_names_out: bool, default=True
            force_int_remainder_cols: bool, default=True
        """
        self.transformations = []
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose_feature_names_out = verbose_feature_names_out
        self.force_int_remainder_cols = force_int_remainder_cols

    def add_transformation(self, transformation: Transformation):
        """Add a transformation to the list of transformations."""
        if not isinstance(transformation, Transformation):
            raise TypeError(
                f"Expected a Transformation object, got {type(transformation)}"
            )
        self.transformations.append(transformation.get_tuple())

    def create_preprocessor(self) -> ColumnTransformer:
        """Create a column transformer preprocessor."""
        return ColumnTransformer(
            transformers=self.transformations,
            remainder=self.remainder,
            sparse_threshold=self.sparse_threshold,
            n_jobs=self.n_jobs,
            transformer_weights=self.transformer_weights,
            verbose_feature_names_out=self.verbose_feature_names_out,
            force_int_remainder_cols=self.force_int_remainder_cols,
        )

    def get_transformations(self) -> List[Tuple[str, Pipeline, List[str]]]:
        """Get the list of transformations."""
        return self.transformations

    def __reduce__(self):
        return (self.__class__, (), self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


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


def engineer_features(
    prep_df_X: pd.DataFrame, transforms: List[Tuple[str, Pipeline, List[str]]]
) -> np.ndarray:
    """
    Applies feature engineering transformations to the provided DataFrame using a column
    preprocessor.

    Args:
        prep_df_X (pd.DataFrame): The input DataFrame containing the features to preprocess.
        transforms (Dict[str, Any]): A dictionary specifying transformations for each column.
            The format of `transforms` should align with the requirements of the
            `create_column_preprocessor` function.

    Returns:
        np.ndarray: The transformed feature matrix after applying preprocessing transformations.
    """
    preprocessor = create_column_preprocessor(transforms)
    transformed_df_X = preprocessor.fit_transform(prep_df_X)
    return transformed_df_X
