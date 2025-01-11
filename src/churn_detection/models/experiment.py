"""Experiment Manager for Churn Detection Models."""

import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from joblib import dump, load
from churn_detection.output_manager import OutputManager
from churn_detection.paths import PARENT_DIR


class ExperimentManager:
    """
    Manages machine learning experiments, including hyperparameter tuning,
    result tracking, and experiment comparison.

    Attributes:
        base_pipeline (BaseEstimator): The base pipeline to use for experiments.
        experiment_name (str): Name of the experiment.
        output_manager (OutputManager): Manages experiment output.
        experiments (Dict[str, Any]): Stores details of all experiments.
        logger (Optional[logging.Logger]): Logger for experiment-related messages.
    """

    def __init__(
        self,
        base_pipeline: BaseEstimator,
        experiment_name: str,
        project_root: Optional[str] = PARENT_DIR,
        enable_logging: bool = True,
        enable_model_saving: bool = False,
    ) -> None:
        self.output_manager = OutputManager(
            project_root=project_root,
            enable_logging=enable_logging,
            enable_model_saving=enable_model_saving,
        )
        self.base_pipeline = base_pipeline
        self.experiment_name = experiment_name
        self.experiments: Dict[str, Any] = {}
        self.logger = (
            logging.getLogger(f"experiment_manager.{experiment_name}")
            if enable_logging
            else None
        )

    def _create_experiment_id(self) -> str:
        """Generates a unique experiment ID based on the current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}"

    def _get_experiment_metadata(self) -> Dict[str, Any]:
        """Retrieves metadata for the experiment."""
        return {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
            "python_version": sys.version,
            "package_versions": {
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }

    def grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
        experiment_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Performs grid search for hyperparameter tuning of both feature engineering
        and model parameters.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            param_grid (Dict[str, List[Any]]): Parameter grid for tuning.
                Can include both model parameters (prefixed with "model__")
                and feature engineering parameters (prefixed with "feature_engineer_N__").
            cv (int): Number of cross-validation folds.
            scoring (str): Scoring metric for evaluation.
            n_jobs (int): Number of parallel jobs.
            experiment_params (Optional[Dict[str, Any]]): Additional parameters.

        Returns:
            Dict[str, Any]: Results of the grid search experiment.
        """
        if self.logger:
            self.logger.info(
                f"Starting grid search for experiment {self.experiment_name}"
            )

        pipeline = self.base_pipeline.get_pipeline()

        # Auto-prefix model parameters if not already prefixed
        prefixed_param_grid = {
            (
                f"model__{param}"
                if not any(
                    param.startswith(prefix)
                    for prefix in ["model__", "feature_engineer_"]
                )
                else param
            ): values
            for param, values in param_grid.items()
        }

        grid_search = GridSearchCV(
            pipeline,
            prefixed_param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=123),
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True,
        )

        if self.logger:
            self.logger.info("Fitting grid search CV")
        grid_search.fit(X_train, y_train)

        # Remove prefixes from best_params for output
        cleaned_best_params = {
            (
                param.replace("model__", "") if param.startswith("model__") else param
            ): value
            for param, value in grid_search.best_params_.items()
        }

        experiment_id = self._create_experiment_id()
        experiment_results = {
            "experiment_id": experiment_id,
            "metadata": self._get_experiment_metadata(),
            "type": "grid_search",
            "best_params": cleaned_best_params,
            "best_score": grid_search.best_score_,
            "cv_results": {
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
                "mean_train_score": grid_search.cv_results_[
                    "mean_train_score"
                ].tolist(),
                "std_train_score": grid_search.cv_results_["std_train_score"].tolist(),
            },
            "param_grid": param_grid,
            "cv": cv,
            "scoring": scoring,
        }

        if experiment_params:
            experiment_results["additional_params"] = experiment_params

        self.experiments[experiment_id] = experiment_results
        self._save_experiment(experiment_results)
        self._save_model(grid_search.best_estimator_, experiment_id)

        if self.logger:
            self.logger.info("Grid search completed.")
        return experiment_results

    def _save_experiment(self, experiment_results: Dict[str, Any]) -> None:
        """Saves experiment results to a file."""
        experiment_id = experiment_results["experiment_id"]
        filepath = self.output_manager.get_experiment_path(experiment_id)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(experiment_results, file, indent=2)

        if self.logger:
            self.logger.info("Experiment saved")

    def _save_model(self, model: BaseEstimator, experiment_id: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            model (BaseEstimator): The model to save.
            experiment_id (str): Unique identifier for the experiment.
        """
        if not self.output_manager.enable_model_saving:
            return

        try:
            model_path = self.output_manager.get_model_path(experiment_id)
            dump(model, model_path)

            if self.logger:
                self.logger.info("Model saved")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, experiment_id: str) -> BaseEstimator:
        """
        Loads a model from a file.

        Args:
            experiment_id (str): ID of the experiment whose model to load.

        Returns:
            BaseEstimator: The loaded model.
        """
        model_path = self.output_manager.get_model_path(experiment_id)
        return load(model_path)

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Loads experiment results from a file."""
        filepath = self.output_manager.get_experiment_path(experiment_id)

        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_best_experiment(self, metric: str = "best_score") -> Dict[str, Any]:
        """
        Retrieves the best experiment based on a specified metric.

        Args:
            metric (str, optional): Metric to evaluate experiments. Defaults to "best_score".

        Returns:
            Dict[str, Any]: Details of the best experiment.

        Raises:
            ValueError: If no experiments have been conducted.
        """
        if not self.experiments:
            raise ValueError("No experiments have been run yet.")

        return max(self.experiments.values(), key=lambda exp: exp[metric])

    def compare_experiments(self, metric: str = "best_score") -> pd.DataFrame:
        """
        Compares all experiments based on a specified metric.

        Args:
            metric (str, optional): Metric to compare experiments. Defaults to "best_score".

        Returns:
            pd.DataFrame: Comparison of experiments with details.

        Raises:
            ValueError: If no experiments have been conducted.
        """
        if not self.experiments:
            raise ValueError("No experiments have been run yet.")

        comparison_data = [
            {
                "experiment_id": exp_id,
                "timestamp": exp["metadata"]["timestamp"],
                metric: exp[metric],
                "parameters": exp["best_params"],
            }
            for exp_id, exp in self.experiments.items()
        ]

        return pd.DataFrame(comparison_data)

    def apply_best_params(self, experiment_id: str) -> None:
        """
        Applies the best parameters from a specific experiment to the base pipeline.

        Args:
            experiment_id (str): ID of the experiment to retrieve parameters from.

        Raises:
            ValueError: If the specified experiment does not exist.
        """
        if experiment_id not in self.experiments:
            experiment = self.load_experiment(experiment_id)
        else:
            experiment = self.experiments[experiment_id]

        best_params = experiment["best_params"]

        # Separate feature engineering and model parameters
        feature_engineer_params = {
            k.replace("feature_engineer_0__", ""): v
            for k, v in best_params.items()
            if k.startswith("feature_engineer_0__")
        }
        model_params = {
            k.replace("model__", ""): v
            for k, v in best_params.items()
            if k.startswith("model__")
        }

        # Apply feature engineering parameters
        feature_engineer = self.base_pipeline.get_pipeline().named_steps[
            "feature_engineer_0"
        ]
        if feature_engineer and hasattr(feature_engineer, "set_params"):
            feature_engineer.set_params(**feature_engineer_params)

        # Apply model parameters
        model = self.base_pipeline.get_pipeline().named_steps["model"]
        if model and hasattr(model, "set_params"):
            model.set_params(**model_params)
