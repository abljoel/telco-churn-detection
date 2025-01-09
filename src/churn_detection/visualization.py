"""
Python Module for Data Visualization

This module provides a variety of functions to generate different types of visualizations for
exploratory data analysis. The functions in this module take in a pandas DataFrame and create
plots using matplotlib or seaborn to help users better understand their data.

Available Functions:
- plot_pie: Plot a pie chart for categorical variable distribution.
- plot_stem: Plot a stem chart for distribution of a specific variable.
- plot_hists: Plot histograms for all columns in the DataFrame.
- plot_shapes: Plot density plots for all columns in the DataFrame.
- plot_boxes: Plot boxplots for all columns in the DataFrame.
- plot_pairs: Plot pairwise relationships for columns in the DataFrame.
- plot_card: Plot bar chart of cardinality for discrete variables.
- plot_cramer: Calculate Cramer's V correlation coefficient between categorical variables.
- get_correlation_info: Plot a heatmap of correlations between numerical columns.
"""

import warnings
from typing import Literal, Any, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import pairplot, heatmap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency


def plot_pie(
    df: pd.DataFrame, catvar: str, title: str = None, size: tuple = (6, 6)
) -> None:
    """
    Plot a pie chart showing the distribution of a categorical variable.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - catvar (str): Column name of the categorical variable to be visualized.
    - title (str, optional): Title of the pie chart. Defaults to a generated title if None.
    - size (tuple, optional): Size of the figure in (width, height). Defaults to (6, 6).

    Returns:
    - None

    Raises:
    - ValueError: If the specified column is not found in the DataFrame or contains no valid values.
    """
    if catvar not in df.columns:
        raise ValueError(f"'{catvar}' column not found in the DataFrame.")
    if df[catvar].isnull().all():
        raise ValueError(f"'{catvar}' column has no valid values to plot.")

    if title is None:
        title = f"Distribution of {catvar} labels"
    plt.figure(figsize=size)
    df[catvar].value_counts(normalize=True).plot.pie(
        autopct="%1.0f%%",
        startangle=90,
    )
    plt.title(title)
    plt.ylabel("")
    plt.show()


def plot_stem(
    df: pd.DataFrame,
    count_var: str,
    title: str = "",
    orientation: str = "vertical",
    size: tuple = (10, 8),
) -> None:
    """
    Plot a stem chart for the distribution of a specified count variable.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - count_var (str): Column name of the variable to plot.
    - title (str, optional): Title of the plot. Defaults to column name if not provided.
    - orientation (str, optional): Orientation of the stem chart, either 'vertical' or 'horizontal'.
    - size (tuple, optional): Size of the figure in (width, height). Defaults to (10, 8).

    Returns:
    - None
    """
    x_vals = df[count_var].value_counts().index
    y_vals = df[count_var].value_counts().values
    plt.figure(figsize=size)
    plt.stem(x_vals, y_vals, linefmt="black", orientation=orientation)
    plt.title(title or count_var)
    plt.show()


def plot_hists(df: pd.DataFrame, size: tuple = (16, 16)) -> None:
    """
    Plot histograms for all numerical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - size (tuple, optional): Size of the figure in (width, height). Defaults to (16, 16).

    Returns:
    - None
    """
    df.hist(figsize=size)
    plt.show()


def plot_shapes(
    df: pd.DataFrame, size: tuple = (12, 7), layout: tuple = (2, 2)
) -> None:
    """
    Plot density (KDE) plots for all numerical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - size (tuple, optional): Size of the figure in (width, height). Defaults to (12, 7).
    - layout (tuple, optional): Grid layout for subplots in (rows, columns). Defaults to (2, 2).

    Returns:
    - None
    """
    axes = df.plot(
        kind="density",
        figsize=size,
        subplots=True,
        layout=layout,
        sharex=False,
        legend=False,
        fontsize=1,
    )
    for ax, col in zip(axes.flatten(), df.columns):
        ax.set_title(col, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_boxes(df: pd.DataFrame, size: tuple = (12, 7)) -> None:
    """
    Plot boxplots for all numerical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - size (tuple, optional): Size of the figure in (width, height). Defaults to (12, 7).

    Returns:
    - None
    """
    plt.figure(figsize=size)
    df.boxplot()
    plt.show()


def plot_pairs(
    df: pd.DataFrame,
    color: str | None = None,
    kind: Literal["scatter", "kde", "hist", "reg"] | None = None,
    height: float = 1.5,
) -> None:
    """
    Plot pairwise relationships between numerical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - color (str, optional): Column name used to color points in pair plot. Defaults to None.
    - kind (str, optional): Type of plot to draw ('scatter', 'kde', etc.). Defaults to None.

    Returns:
    - None
    """
    pairplot(df, hue=color, kind=kind or "scatter", height=height)
    plt.show()


def plot_card(
    df: pd.DataFrame, title: str = "Cardinality of Discrete Variables"
) -> None:
    """
    Plot a bar chart representing the cardinality (number of unique values) for discrete variables.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - title (str, optional): Title of the plot. Defaults to "Cardinality of Discrete Variables".

    Returns:
    - None
    """
    df.nunique().sort_values().plot.bar()
    plt.title(title)
    plt.show()


def plot_bivariate_cat(df: pd.DataFrame, x: str, y: str, title: str = "") -> None:
    """
    Plot a bivariate count plot for two categorical variables.

    This function creates a count plot for the given categorical variables to visualize
    the frequency distribution between them.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical data.
    - x (str): The column name of the first categorical variable (x-axis).
    - y (str): The column name of the second categorical variable used for hue.
    - title (str, optional): The title of the plot. Defaults to a generated title if not provided.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=x, hue=y)
    plt.title(title if title else f"{x} vs. {y}", fontsize=16)
    plt.xlabel(f"{x}")
    plt.ylabel("Count")
    plt.legend(title=f"{y}")
    plt.tight_layout()
    plt.show()


def plot_correlation_info(
    df: pd.DataFrame,
    size: tuple = (8, 6),
    method: str = "pearson",
    coefmin: float = -1,
    coefmax: float = 1,
) -> None:
    """
    Plot a heatmap showing correlations between numerical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - size (tuple, optional): Size of the figure in (width, height). Defaults to (8, 6).
    - method (str, optional): Correlation method ('pearson', 'spearman', 'kendall').
                              Defaults to "pearson".
    - coefmin (float, optional): Minimum value for color scale. Defaults to -1.
    - coefmax (float, optional): Maximum value for color scale. Defaults to 1.

    Returns:
    - None
    """
    plt.figure(figsize=size)
    heatmap(df.corr(method=method), annot=True, vmin=coefmin, vmax=coefmax)
    plt.title("Variable Correlation Heatmap")
    plt.show()


def get_cramer_v(label: pd.Series, x: pd.Series, bias_correction: bool = True) -> float:
    """
    Calculate Cramer's V correlation coefficient between two categorical variables.

    Parameters:
    - label (pd.Series): The first categorical variable.
    - x (pd.Series): The second categorical variable.
    - bias_correction (bool, optional): Whether to apply bias correction to the coefficient.
      Defaults to True.

    Returns:
    - float: The Cramer's V correlation coefficient.

    Raises:
    - ValueError: If the input series have incompatible dimensions.
    """
    if label.shape[0] != x.shape[0]:
        raise ValueError("Input series must have the same length.")

    confusion_table = pd.crosstab(label, x)
    chi2 = chi2_contingency(confusion_table)[0]
    n = confusion_table.sum().sum()
    r, k = confusion_table.shape
    phi2 = chi2 / n

    if bias_correction:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if min((kcorr - 1), (rcorr - 1)) == 0:
            warnings.warn(
                "Unable to calculate Cramer's V using bias correction. Consider not "
                "using bias correction",
                RuntimeWarning,
            )
            return 0.0
        else:
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        return np.sqrt(phi2 / min(k - 1, r - 1))


def plot_cramer(df: pd.DataFrame, size: tuple = (12, 10)) -> None:
    """
    Plot a heatmap of Cramer's V correlation coefficients between all categorical columns
    in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical data to be analyzed.

    Returns:
    - None
    """
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    cramer_table = pd.DataFrame(
        index=categorical_columns, columns=categorical_columns, dtype=float
    )

    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 == col2:
                cramer_table.loc[col1, col2] = 1.0
            else:
                cramer_table.loc[col1, col2] = get_cramer_v(df[col1], df[col2])

    plt.figure(figsize=size)
    sns.heatmap(
        cramer_table, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5
    )
    plt.title("Cramer's V Correlation Heatmap")
    plt.show()


def plot_roc_curve(
    model: Any,
    X: Any,
    y: Any,
    average: Literal["micro", "macro", "samples", "weighted"] | None = "macro",
) -> None:
    """
    Plot the ROC curve for a given model and dataset.

    Parameters:
    model (Any): Trained model that has a `predict_proba` method.
    X (Any): Feature dataset, typically a NumPy array or pandas DataFrame.
    y (Any): True labels, typically a NumPy array or pandas Series.

    Returns:
    None
    """
    y_probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = roc_auc_score(y, y_probs, average=average)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_confusion_table(
    y: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> None:
    """
    Display the confusion matrix for given predictions and true labels.

    Args:
        y (Union[np.ndarray, pd.Series]): True labels as a NumPy array or pandas Series.
        y_pred (Union[np.ndarray, pd.Series]): Predicted labels as a NumPy array or pandas Series.

    Raises:
        ValueError: If `y` or `y_pred` are empty.
    """
    if y is None or len(y) == 0:
        raise ValueError("The target labels 'y' cannot be empty.")

    if y_pred is None or len(y_pred) == 0:
        raise ValueError("The predicted labels 'y_pred' cannot be empty.")

    ConfusionMatrixDisplay(confusion_matrix(y, y_pred, normalize="true")).plot()
    plt.show()


def churn_performance_report(
    y_true: Union[np.ndarray, list, pd.Series],
    y_pred: Union[np.ndarray, list],
    y_pred_proba: Optional[Union[np.ndarray, list]] = None,
) -> None:
    """
    Generate a performance evaluation report for a churn prediction model.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground truth labels.
        y_pred (Union[np.ndarray, list]): Predicted labels.
        y_pred_proba (Optional[Union[np.ndarray, list]]): Predicted probabilities for the
                                                          positive class.
                                                          Required for ROC-AUC and Precision-Recall
                                                          Curve.

    Returns:
        None: Displays plots and prints evaluation metrics.
    """
    if not isinstance(y_true, (np.ndarray, list, pd.Series)):
        raise ValueError(
            "y_true must be array-like (e.g., list, numpy array, or pandas Series)."
        )

    if not isinstance(y_pred, (np.ndarray, list)):
        raise ValueError("y_pred must be array-like (e.g., list or numpy array).")

    if y_pred_proba is not None and not isinstance(y_pred_proba, (np.ndarray, list)):
        raise ValueError("y_pred_proba must be array-like if provided.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    plt.figure(figsize=(12, 10))
    plt.suptitle("Churn Prediction Performance Report", fontsize=16, y=1.02)

    # Confusion Matrix with Normalization
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Churned", "Churned"]
    )
    disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
    plt.title("Confusion Matrix (Normalized)")

    # Precision-Recall Curve
    if y_pred_proba is not None:
        plt.subplot(2, 2, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
        plt.fill_between(recall, precision, alpha=0.2, color="b")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

    # Pie Chart: Percentage of Churned Customers Reached
    plt.subplot(2, 2, 3)
    churned_reached = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    total_churned = np.sum(np.array(y_true) == 1)
    if total_churned > 0:
        labels = ["Reached", "Not Reached"]
        sizes = [churned_reached, total_churned - churned_reached]
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=["lightcoral", "lightblue"],
            startangle=90,
        )
        plt.title("Percentage of Churned Customers Reached")
    else:
        plt.text(
            0.5, 0.5, "No Churned Customers in Ground Truth", ha="center", va="center"
        )

    # Evaluation Metrics
    plt.subplot(2, 2, 4)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["ROC-AUC Score"] = roc_auc_score(y_true, y_pred_proba)
        metrics["Average Precision Score"] = average_precision_score(
            y_true, y_pred_proba
        )

    metrics_text = "\n".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
    plt.text(0.2, 0.6, metrics_text, fontsize=12, va="center")
    plt.axis("off")
    plt.title("Evaluation Metrics")

    plt.tight_layout()
    plt.show()
