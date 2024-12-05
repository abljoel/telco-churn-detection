"""Python Module for data utilities.

This module provides utility functions for analyzing and summarizing pandas DataFrames,
including functions for finding duplicate rows and displaying metadata summaries.
"""

from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from .paths import TRANSFORMED_DATA_DIR


VALID_DATASET_NAMES = {"train", "validation", "test"}


def load_datasets(
    *, names: List[str], as_df: bool = True
) -> Union[List[pd.DataFrame], List[np.ndarray]]:
    """
    Load datasets from feather files.

    Args:
        names (List[str]): A list of dataset names. Must be one or more of 'train', 'validation',
                           and 'test'.
        as_df (bool, optional): If True, returns a list of DataFrames. If False, returns a list
                                of NumPy arrays.

    Returns:
        Union[List[pd.DataFrame], List[np.ndarray]]: Loaded datasets as either DataFrames or NumPy
                                                     arrays.

    Raises:
        ValueError: If no dataset names are provided or if any name is invalid.
    """
    if not names:
        raise ValueError("No dataset name(s) provided.")

    invalid_names = [name for name in names if name not in VALID_DATASET_NAMES]
    if invalid_names:
        raise ValueError(
            f"Invalid dataset name(s) provided: {', '.join(invalid_names)}. Valid names are: "
            f"{', '.join(VALID_DATASET_NAMES)}"
        )

    datasets = [
        pd.read_feather(TRANSFORMED_DATA_DIR / f"{name}.feather") for name in names
    ]

    if as_df:
        return datasets
    return [df.values for df in datasets]


def get_duplicate_count(df: pd.DataFrame) -> int:
    """
    Returns the count of duplicated rows in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze for duplicates.

    Returns:
        int: The number of duplicated rows in the DataFrame.
    """
    return df[df.duplicated(keep=False)].shape[0]


def get_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns all duplicated rows in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze for duplicates.

    Returns:
        pd.DataFrame: A DataFrame containing all duplicated rows.
    """
    return df[df.duplicated(keep=False)]


def get_duplicates(df: pd.DataFrame, return_type: str = "count") -> Any:
    """
    Returns either the count of duplicated rows or the duplicated rows themselves, depending on
    the specified return type.

    Args:
        df (pd.DataFrame): The DataFrame to analyze for duplicates.
        return_type (str): Specifies what to return. Options are 'count' for the number of
                           duplicated rows, or 'rows' for the actual duplicated rows.
                           Defaults to 'count'.

    Returns:
        Any: If `return_type` is 'count', returns an integer representing the number of
             duplicated rows.
             If `return_type` is 'rows', returns a DataFrame containing the duplicated rows.

    Raises:
        ValueError: If `return_type` is not 'count' or 'rows'.
    """
    if return_type == "count":
        return get_duplicate_count(df)
    if return_type == "rows":
        return get_duplicate_rows(df)
    raise ValueError("Invalid return_type. Supported values are 'count' or 'rows'.")


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a summary of metadata from a DataFrame.

    This function returns key information about the DataFrame, including:
    - Dataset dimensions (number of rows and columns)
    - Column names
    - Data types (either detailed or summarized if many columns exist)
    - Cardinality (unique values per column)
    - Sample of values for each column
    - Missing values information
    - Count of duplicated rows

    Args:
        df (pd.DataFrame): The DataFrame to summarize.

    Returns:
        Dict[str, Any]: A dictionary containing the metadata summary of the DataFrame.
    """
    info = {
        "dimensions": (df.shape[0], df.shape[1]),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes if df.shape[1] < 84 else df.dtypes.unique().tolist(),
        "cardinality": df.nunique().sort_values().to_dict(),
        "sample_values": {col: sorted(df[col].value_counts().index[:5]) for col in df},
        "missing_values": (
            (df.isna().mean() * 100).to_dict()
            if df.isna().sum().size < 64
            else "Too many columns with NA values"
        ),
        "duplicate_count": get_duplicate_count(df),
    }
    return info


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Display the dataset information in a readable format.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.

    Returns:
        None: This function only prints information and does not return a value.
    """
    info = get_dataset_info(df)
    print(
        f"Dataset dimensions: {info['dimensions'][0]} rows and {info['dimensions'][1]} columns"
    )
    print("-----------------------------------")
    print("Attribute set:")
    print(info["columns"])
    print("-----------------------------------")
    print("Data types:")
    print(info["data_types"])
    print("-----------------------------------")
    print("Cardinality in variables:")
    for k, v in info["cardinality"].items():
        print(f"{k} -> {v}")
    print("-----------------------------------")
    print("Values in variables:")
    for col, values in info["sample_values"].items():
        print(f"{col} -> {' '.join(str(val) for val in values)} ...")
    print("-----------------------------------")
    print("Missing values in %:")
    if isinstance(info["missing_values"], str):
        print(info["missing_values"])
    else:
        for k, v in info["missing_values"].items():
            print(f"{k} -> {v}")
    print("-----------------------------------")
    print(f"Number of duplicated rows: {info['duplicate_count']}")


def get_distribution_info(df: pd.DataFrame) -> None:
    """
    Provides a comprehensive analysis of the distribution of data in a DataFrame.

    This function prints out various statistical summaries and distribution analyses
    for the numeric columns in the given DataFrame. It includes mean, median, standard
    deviation, minimum, maximum, dispersion coefficients, kurtosis, and skewness.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
        None
    """
    numeric_cols = df.select_dtypes("number")
    print("-----------------------------------")
    print("Distribution analysis:")
    print(numeric_cols.describe().T[["mean", "50%", "std"]])
    print("-----------------------------------")
    print("Variable ranges:")
    print(numeric_cols.describe().T[["min", "max"]])
    print("-----------------------------------")
    print("Quartile dispersion coefficients:")
    threshold = 10
    target_cols = [
        col for col, count in numeric_cols.nunique().items() if count > threshold
    ]
    for var in target_cols:
        var_q = numeric_cols[var].quantile(q=[0.25, 0.75]).tolist()
        try:
            var_qcod = (var_q[1] - var_q[0]) / (var_q[0] + var_q[1])
        except ZeroDivisionError:
            var_qcod = 0.0
        print(f"{var} -> {var_qcod:.2f}")
    print("-----------------------------------")
    print("Kurtosis:")
    print(numeric_cols.kurtosis().abs())
    print("-----------------------------------")
    print("Skewness:")
    print(numeric_cols.skew().abs())
    print()


def get_feature_names(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Get the names of numeric and categorical variables from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        tuple[list[str], list[str]]: A tuple containing a list of numeric variable names and a
                                     list of categorical variable names.
    """
    numeric_variables = ["tenure", "monthlycharges", "totalcharges"]

    categorical_variables = [
        var for var in df.columns
        if var not in numeric_variables
        and var != "churn"
    ]
    return numeric_variables, categorical_variables
