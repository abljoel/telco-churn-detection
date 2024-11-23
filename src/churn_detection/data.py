"""
This module handles the batch download, extraction, and saving of customer churn data
from a Kaggle dataset for further analysis.

The script provides utility functions to:
- Fetch data from Kaggle and extract it from a zip file.
- Save the data in the specified format (Feather or CSV) for optimized storage and retrieval.
"""

import subprocess
import zipfile
import os
from typing import NoReturn

import pandas as pd


def fetch_batch_data(
    target: str,
    cwd_path: str | os.PathLike,
    zip_file: str | os.PathLike,
    raw_data: str | os.PathLike,
) -> pd.DataFrame:
    """
    Downloads the specified Kaggle dataset, extracts the CSV file from the zip archive,
    and loads it into a pandas DataFrame.

    Args:
        target (str): The Kaggle dataset identifier to fetch, e.g. 'blastchar/telco-customer-churn'.
        cwd_path (str | os.PathLike): The path to the CWD where files will be extracted.
        zip_file (str | os.PathLike): The path to the zip file to be downloaded.
        raw_data (str | os.PathLike): The path to the extracted CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted customer churn data.
    """
    subprocess.run(["kaggle", "datasets", "download", "-d", target], check=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(cwd_path)
    data = pd.read_csv(raw_data)
    return data


def save_batch_data(
    df: pd.DataFrame,
    target_path: str | os.PathLike,
    zip_file: str | os.PathLike,
    raw_data: str | os.PathLike,
    file_format: str = "feather",
) -> NoReturn:
    """
    Saves the given DataFrame to the specified file format (Feather or CSV) and deletes
    temporary files used during the data download and extraction process.

    Args:
        df (pd.DataFrame): The customer churn DataFrame to save.
        target_path (str | os.PathLike): The path where the DataFrame should be saved.
        zip_file (str | os.PathLike): The path to the zip file that was downloaded.
        raw_data (str | os.PathLike): The path to the raw CSV file that was extracted.
        file_format (str): The format in which to save the DataFrame. Supported formats 
                           are 'csv' or 'feather'. Defaults to 'feather'.

    Raises:
        ValueError: If the provided file format is not 'csv' or 'feather'.

    Side Effects:
        - Saves the DataFrame in the specified format in the target directory.
        - Deletes the downloaded zip and CSV files from the specified locations.
    """
    if file_format in ("csv", "feather"):
        if file_format == "feather":
            df.to_feather(target_path / "customer_churn.feather")
        else:
            df.to_csv(target_path / "customer_churn.csv", index=None)
        subprocess.run(["cmd", "/c", "del", str(zip_file)], check=True)
        subprocess.run(["cmd", "/c", "del", str(raw_data)], check=True)
    else:
        raise ValueError(
            f"Invalid file format '{file_format}'. Supported formats are 'csv' or 'feather'."
        )
