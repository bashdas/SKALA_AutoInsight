"""Weights & Biases and Weave initialization helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import wandb


DEFAULT_ENTITY = "skala-autoinsight"
DEFAULT_PROJECT = "autoinsight_mini_project"


def project_path() -> str:
    """Return the configured ``entity/project`` path."""
    entity = os.getenv("WANDB_ENTITY", DEFAULT_ENTITY)
    project = os.getenv("WANDB_PROJECT", DEFAULT_PROJECT)
    return f"{entity}/{project}"


def init_run(
    *,
    name: str | None = None,
    job_type: str | None = None,
    config: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    mode: str | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run using the project's shared defaults."""
    entity, project = project_path().split("/", maxsplit=1)
    return wandb.init(
        entity=entity,
        project=project,
        name=name,
        job_type=job_type,
        config=dict(config or {}),
        tags=list(tags or []),
        mode=mode,
    )


def init_weave() -> None:
    """Initialize Weave for operations that need trace logging."""
    import weave

    weave.init(project_path())
