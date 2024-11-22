"""
This module handles the batch download, extraction, and saving of customer churn data
from a Kaggle dataset for further analysis.

The script provides utility functions to:
- Fetch data from Kaggle and extract it from a zip file.
- Save the data in Feather format for optimized storage and retrieval.
"""

import subprocess
import zipfile
from pathlib import Path
from typing import NoReturn

import pandas as pd

from churn_detection.paths import EXTERNAL_DATA_DIR

CURRENT_DIR = Path().cwd()
ZIP_FILE = Path("telco-customer-churn.zip")
CSV_FILE = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")


def fetch_batch_data(target: str) -> pd.DataFrame:
    """
    Downloads the specified Kaggle dataset, extracts the CSV file from the zip archive,
    and loads it into a pandas DataFrame.

    Args:
        target (str): The Kaggle dataset to download, e.g. 'blastchar/telco-customer-churn'.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted customer churn data.
    """
    subprocess.run(["kaggle", "datasets", "download", "-d", target], check=True)
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(CURRENT_DIR)
    data = pd.read_csv(CSV_FILE)
    return data


def save_batch_data(df: pd.DataFrame, file_format: str = "feather") -> NoReturn:
    """
    Saves the given DataFrame to the specified file format (Feather or CSV) and deletes
    temporary files.

    Args:
        df (pd.DataFrame): The customer churn DataFrame to save.
        file_format (str): The format in which to save the DataFrame. Supported formats
        are 'csv' or 'feather'.
                           Defaults to 'feather'.

    Raises:
        ValueError: If the provided file format is not 'csv' or 'feather'.

    Side Effects:
        - Saves the DataFrame in the specified format in the external data directory.
        - Deletes the downloaded zip and CSV files from the current directory.
    """
    if file_format in ("csv", "feather"):
        if file_format == "feather":
            df.to_feather(EXTERNAL_DATA_DIR / "customer_churn.feather")
        else:
            df.to_csv(EXTERNAL_DATA_DIR / "customer_churn.csv", index=None)
        subprocess.run(["cmd", "/c", "del", str(ZIP_FILE)], check=True)
        subprocess.run(["cmd", "/c", "del", str(CSV_FILE)], check=True)
    else:
        raise ValueError(
            f"Invalid file format '{file_format}'. Supported formats are 'csv' or 'feather'."
        )
