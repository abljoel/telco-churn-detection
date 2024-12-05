"""Processing Utilities Module

This module handles the preprocessing step before feature engineering and model training.
"""

from typing import Tuple, Union
import pandas as pd


def clean_data(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Cleans the input DataFrame by transforming columns to lower case,
    mapping 'Churn' column to numerical values, and handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    new_df = df.copy()

    if target in new_df.columns:
        new_df[target] = new_df[target].map({"Yes": 1, "No": 0})

    new_df.columns = new_df.columns.str.lower()

    if "customerid" in new_df.columns:
        new_df.drop(columns="customerid", inplace=True)

    if "totalcharges" in new_df.columns:
        new_df["totalcharges"] = pd.to_numeric(new_df["totalcharges"], errors="coerce")
        new_df["totalcharges"] = new_df["totalcharges"].fillna(0)

    for col in new_df.select_dtypes(include="object").columns:
        new_df[col] = new_df[col].str.lower().str.replace(" ", "_")

    return new_df


def split_data(
    df: pd.DataFrame, target: str = "churn"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits the DataFrame into features and target.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        target (str, optional): The target column name. Defaults to 'churn'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature DataFrame (X) and the
                                        target Series (y).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    features = df.drop(columns=target)
    targets = df[target]

    return features, targets


def preprocess_data(
    raw_data: pd.DataFrame, split: bool = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """Preprocesses the raw data by cleaning it and then optionally splitting it into features
    and target.

    Args:
        raw_data (pd.DataFrame): The input raw DataFrame.
        split (bool, optional): Whether to split the data into features and target. 
                                Defaults to None.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]: The cleaned DataFrame or a tuple
                                                             containing the feature DataFrame (X)
                                                             and the target Series (y).
    """
    cleaned_data = clean_data(raw_data)
    if split:
        return split_data(cleaned_data)

    return cleaned_data
