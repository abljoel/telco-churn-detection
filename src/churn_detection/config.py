"""Configuration module for loading YAML configuration files."""

from typing import Any, Dict, Union
from pathlib import Path
import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path (Union[str, Path]): The file path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration data loaded from the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("The 'path' argument must be a string or a Path object.")

    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at '{path}' was not found.") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"An error occurred while parsing the YAML file at '{path}'."
        ) from e
