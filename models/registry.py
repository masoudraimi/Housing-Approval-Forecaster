"""MLflow model registration and alias promotion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.pytorch

__all__ = ["register_model", "promote_to_champion", "load_champion"]

_DEFAULT_CHAMPION_ALIAS = "champion"


def register_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    metrics: Optional[dict] = None,
    tags: Optional[dict] = None,
) -> str:
    """Register a logged model artefact in the MLflow model registry.

    Returns the model URI of the registered version.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version

    client = mlflow.tracking.MlflowClient()
    if metrics:
        for key, value in metrics.items():
            client.set_model_version_tag(model_name, version, key, str(value))
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(model_name, version, key, str(value))

    print(f"Registered {model_name} version {version} from run {run_id}")
    return f"models:/{model_name}/{version}"


def promote_to_champion(model_name: str, version: str) -> None:
    """Set the champion alias on a registered model version."""
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, _DEFAULT_CHAMPION_ALIAS, version)
    print(f"Promoted {model_name} v{version} to @{_DEFAULT_CHAMPION_ALIAS}")


def load_champion(model_name: str) -> Any:
    """Load the model currently tagged as champion from the MLflow registry."""
    uri = f"models:/{model_name}@{_DEFAULT_CHAMPION_ALIAS}"
    return mlflow.pytorch.load_model(uri)
