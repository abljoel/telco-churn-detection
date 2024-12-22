"""Experiment Manager for Churn Detection Models."""

import json
from datetime import datetime
import os
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ..paths import EXPERIMENT_DIR


class ExperimentManager:
    """
    A class to manage machine learning experiments, including:
    - Hyperparameter tuning
    - Result tracking
    - Experiment comparison
    """

    def __init__(
        self, base_pipeline: Any, experiment_name: str, save_dir: str = EXPERIMENT_DIR
    ):
        """
        Initialize the ExperimentManager.

        Args:
            base_pipeline (Any): The base pipeline containing feature engineers and model.
            experiment_name (str): Name of the experiment.
            save_dir (str): Directory to save experiment results.
        """
        self.base_pipeline = base_pipeline
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self._setup_save_directory()

    def _setup_save_directory(self) -> None:
        """Create directory for saving experiment results."""
        os.makedirs(self.save_dir, exist_ok=True)

    def _create_experiment_id(self) -> str:
        """Create unique experiment ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}"

    def _add_param_prefix(self, params: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add prefix to parameter names if not already present.

        Args:
            params (Dict[str, Any]): Original parameter dictionary.
            prefix (str): Prefix to add.

        Returns:
            Dict[str, Any]: Updated parameter dictionary with prefixes.
        """
        return {
            (f"{prefix}__{key}" if not key.startswith(f"{prefix}__") else key): value
            for key, value in params.items()
        }

    def _strip_param_prefix(
        self, params: Dict[str, Any], prefix: str
    ) -> Dict[str, Any]:
        """Remove prefix from parameter names.

        Args:
            params (Dict[str, Any]): Parameter dictionary with prefixes.
            prefix (str): Prefix to remove.

        Returns:
            Dict[str, Any]: Updated parameter dictionary without prefixes.
        """
        prefix_str = f"{prefix}__"
        return {
            (key[len(prefix_str) :] if key.startswith(prefix_str) else key): value
            for key, value in params.items()
        }

    def grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, List[Any]],
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
        experiment_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): Training data.
            y_train (Union[pd.Series, List[Any]]): Target values.
            param_grid (Dict[str, List[Any]]): Parameters to tune.
            cv (int): Number of cross-validation folds.
            scoring (str): Metric to optimize.
            n_jobs (int): Number of parallel jobs.
            experiment_params (Optional[Dict[str, Any]]): Additional parameters to track
                                                          with the experiment.

        Returns:
            Dict[str, Any]: Experiment results.
        """
        steps = [
            (f"feature_engineer_{i}", fe)
            for i, fe in enumerate(self.base_pipeline.feature_engineers)
        ]
        steps.append(("model", self.base_pipeline.model))
        pipeline = Pipeline(steps=steps)

        prefixed_param_grid = self._add_param_prefix(param_grid, "model")

        grid_search = GridSearchCV(
            pipeline,
            prefixed_param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)

        best_params = self._strip_param_prefix(grid_search.best_params_, "model")

        experiment_id = self._create_experiment_id()
        experiment_results = {
            "experiment_id": experiment_id,
            "type": "grid_search",
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
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

        return experiment_results

    def _save_experiment(self, experiment_results: Dict[str, Any]) -> None:
        """Save experiment results to file."""
        filepath = os.path.join(
            self.save_dir, f"{experiment_results['experiment_id']}.json"
        )
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(experiment_results, file, indent=2)

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment results from file.

        Args:
            experiment_id (str): ID of the experiment to load.

        Returns:
            Dict[str, Any]: Loaded experiment results.
        """
        filepath = os.path.join(self.save_dir, f"{experiment_id}.json")
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_best_experiment(self, metric: str = "best_score") -> Dict[str, Any]:
        """Get the best experiment based on a specified metric.

        Args:
            metric (str): Metric to compare experiments.

        Returns:
            Dict[str, Any]: Best experiment results.

        Raises:
            ValueError: If no experiments have been run.
        """
        if not self.experiments:
            raise ValueError("No experiments have been run yet.")

        return max(self.experiments.values(), key=lambda x: x[metric])

    def compare_experiments(self, metric: str = "best_score") -> pd.DataFrame:
        """Compare all experiments based on a specified metric.

        Args:
            metric (str): Metric to compare experiments.

        Returns:
            pd.DataFrame: DataFrame containing comparison results.

        Raises:
            ValueError: If no experiments have been run.
        """
        if not self.experiments:
            raise ValueError("No experiments have been run yet.")

        comparison_data = [
            {
                "experiment_id": exp_id,
                "timestamp": exp_results["timestamp"],
                metric: exp_results[metric],
                "parameters": exp_results["best_params"],
            }
            for exp_id, exp_results in self.experiments.items()
        ]

        return pd.DataFrame(comparison_data)

    def apply_best_params(self, experiment_id: str) -> None:
        """Apply the best parameters from a specific experiment to the base pipeline.

        Args:
            experiment_id (str): ID of the experiment to apply.

        Raises:
            ValueError: If the specified experiment ID is not found.
        """
        if experiment_id not in self.experiments:
            experiment = self.load_experiment(experiment_id)
        else:
            experiment = self.experiments[experiment_id]

        best_params = experiment["best_params"]
        self.base_pipeline.model.set_params(**best_params)
