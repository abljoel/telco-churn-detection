"""Init file for churn_detection"""

from .config import load_config
from .data import load_data, fetch_batch_data, save_batch_data
from .preprocessing import preprocess_data, clean_data, split_data
from .evaluation import (
    validate_model_with_cv,
    display_roc_auc_score,
    display_pr_auc_score,
    display_clf_report,
    explore_thresholds,
)
from .features import (
    FeatureConcatenator,
    RareCategoryEncoder,
    InteractionStrengthExtractor,
    Transformation,
    ColumnPreprocessor,
)
from .utils import (
    load_datasets,
    get_duplicates,
    get_dataset_info,
    display_dataset_info,
    get_distribution_info,
    get_feature_names,
)
from .visualization import (
    plot_pie,
    plot_correlation_info,
    plot_bivariate_cat,
    plot_confusion_table,
    churn_performance_report,
)
from .models import (
    LinearModel,
    SklearnModel,
    ExperimentManager,
    create_pipeline,
    MLPipeline,
)
from .models.base_model import (
    BaseModel,
)  # Direct import since it's used by multiple modules

__version__ = "0.1.0"

__all__ = [
    # Data Loading & Config
    "load_config",
    "load_data",
    "fetch_batch_data",
    "save_batch_data",
    # Preprocessing
    "preprocess_data",
    "clean_data",
    "split_data",
    # Evaluation
    "validate_model_with_cv",
    "display_roc_auc_score",
    "display_pr_auc_score",
    "display_clf_report",
    "explore_thresholds",
    # Feature Engineering
    "FeatureConcatenator",
    "RareCategoryEncoder",
    "InteractionStrengthExtractor",
    "Transformation",
    "ColumnPreprocessor",
    # Utilities
    "load_datasets",
    "get_duplicates",
    "get_dataset_info",
    "display_dataset_info",
    "get_distribution_info",
    "get_feature_names",
    # Visualization
    "plot_pie",
    "plot_correlation_info",
    "plot_bivariate_cat",
    "plot_confusion_table",
    "churn_performance_report",
    # Models
    "BaseModel",
    "LinearModel",
    "SklearnModel",
    "ExperimentManager",
    "create_pipeline",
    "MLPipeline",
]
