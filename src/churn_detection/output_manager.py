"""Output management utilities for ML experiments."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class OutputManager:
    """Manages output directory structure for ML experiments."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        enable_logging: bool = True,
        enable_model_saving: bool = True,
    ) -> None:
        """
        Initialize the OutputManager with optional project root and feature toggles.

        Args:
            project_root (Optional[str]): Root directory for the project. Defaults to the current 
                                          working directory.
            enable_logging (bool): Whether to enable logging. Defaults to True.
            enable_model_saving (bool): Whether to enable model saving. Defaults to True.
        """
        self.project_root: Path = self._get_project_root(project_root)
        self.outputs_dir: Path = self.project_root / "outputs"
        self.experiments_dir: Path = self.outputs_dir / "experiments"
        self.logs_dir: Path = self.outputs_dir / "logs"
        self.models_dir: Path = self.outputs_dir / "models"
        self.enable_logging = enable_logging
        self.enable_model_saving = enable_model_saving
        self._setup_directories()

        if self.enable_logging:
            self._setup_logging()

    def _get_project_root(self, project_root: Optional[str]) -> Path:
        """
        Resolve the project root directory.

        Args:
            project_root (Optional[str]): Root directory path as a string. Defaults to None.

        Returns:
            Path: Resolved project root as a Path object.
        """
        return Path(project_root) if project_root else Path.cwd()

    def _setup_directories(self) -> None:
        """
        Create necessary output directories if they don't already exist.
        """
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_logging:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_model_saving:
            self.models_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """
        Configure logging to output to a file and the console.
        """
        log_file: Path = self.logs_dir / f"experiments_{datetime.now():%Y%m%d}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def get_experiment_path(self, experiment_id: str) -> Path:
        """
        Retrieve the path for storing experiment results.

        Args:
            experiment_id (str): Unique identifier for the experiment.

        Returns:
            Path: Path to the experiment results file.
        """
        return self.experiments_dir / f"{experiment_id}.json"

    def get_model_path(self, experiment_id: str) -> Path:
        """
        Retrieve the path for saving model artifacts.

        Args:
            experiment_id (str): Unique identifier for the experiment.

        Returns:
            Path: Path to the model artifact file.

        Raises:
            ValueError: If model saving is disabled.
        """
        if not self.enable_model_saving:
            raise ValueError("Model saving is disabled.")
        return self.models_dir / f"{experiment_id}_model.joblib"
