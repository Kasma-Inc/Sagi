import json
import os
import re
from typing import Any, Dict, Union

import tomli


def _replace_env_vars(field: str) -> str:
    """
    Replace environment variables in a string with their values from the environment.

    Args:
        field: A string to replace environment variables in, and the environment variables
        should be in the format of ${VAR} or $VAR:DEFAULT_VALUE.

    Returns:
        The string with environment variables replaced with their values from the environment.

    Examples:
        >>> os.environ["LLAMA_BASE_URL"] = "http://localhost:8000"
        >>> _replace_env_vars("${LLAMA_BASE_URL}")
        'http://localhost:8000'
        >>> _replace_env_vars("$LLAMA_BASE_URL")
        'http://localhost:8000'
    """
    pattern = r"\$(?:{[A-Za-z_][A-Za-z0-9_]*}|[A-Za-z_][A-Za-z0-9_]*)"

    def replace_match(match):
        var = match.group(0)
        var_name = var.strip("${}").lstrip("$")
        try:
            return os.environ.get(var_name)
        except Exception as e:
            print(f"Warning: {var_name} not found in environment variables")
            return var

    return re.sub(pattern, replace_match, field)


def replace_env_vars_in_dict(field: Union[Dict[str, Any], str]) -> Any:
    """
    Replace environment variables in a dictionary or string with their values from the configs dictionary.

    Args:
        field: A dictionary or string to replace environment variables in.
        configs: A dictionary of environment variable names and their values.

    Returns:
        The field with environment variables replaced with their values from the configs dictionary.

    Examples:
        >>> os.environ["LLAMA_BASE_URL"] = "http://localhost:8000"
        >>> os.environ["LLAMA_API_KEY"] = "1234567890"
        >>> field = {
        ...     "base_url_env_var": "${LLAMA_BASE_URL}",
        ...     "api_key": "${LLAMA_API_KEY}",
        ... }
        >>> replace_env_vars_in_dict(field)
        {'base_url_env_var': 'http://localhost:8000', 'api_key': '1234567890'}
    """
    if isinstance(field, dict):
        return {k: replace_env_vars_in_dict(v) for k, v in field.items()}
    elif isinstance(field, str):
        return _replace_env_vars(field)

    return field


def read_json_with_env_vars(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file {file_path}: {e}")

    return replace_env_vars_in_dict(data)


def load_toml_with_env_vars(file_path: str):
    try:
        with open(file_path, "rb") as f:
            data = tomli.load(f)
    except Exception as e:
        raise ValueError(f"Error loading TOML file {file_path}: {e}")

    return replace_env_vars_in_dict(data)
