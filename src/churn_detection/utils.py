"""Python Module for data utilities.

This module provides utility functions for analyzing and summarizing pandas DataFrames,
including functions for finding duplicate rows and displaying metadata summaries.
"""

from typing import Dict, Any
import pandas as pd


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


# Example of how to use the updated `get_dataset_info` function
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
