# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Loong Ma

"""Registry for built-in dataset builders."""

from __future__ import annotations

from .agent_trajectory import AgentTrajectoryDatasetBuilder
from .base import BuiltinDatasetBuilder, DatasetBuilderError
from .skill_query import SkillQueryDatasetBuilder

_DATASET_BUILDERS: dict[str, BuiltinDatasetBuilder] = {
    AgentTrajectoryDatasetBuilder.dataset_type: AgentTrajectoryDatasetBuilder(),
    SkillQueryDatasetBuilder.dataset_type: SkillQueryDatasetBuilder(),
}


def get_dataset_builder(dataset_type: str) -> BuiltinDatasetBuilder:
    """Return the registered builder for a dataset type."""

    try:
        return _DATASET_BUILDERS[dataset_type]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASET_BUILDERS))
        raise DatasetBuilderError(f"Unknown dataset type '{dataset_type}'. Available types: {known}") from exc
