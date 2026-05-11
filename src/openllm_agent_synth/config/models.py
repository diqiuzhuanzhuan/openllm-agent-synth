# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Loong Ma

"""Pydantic models for YAML-driven application configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_TASK_DOMAINS = {
    "tool_calling": [
        "calendar_management",
        "travel_planning",
        "crm_lookup",
        "support_triage",
        "research_lookup",
    ],
    "code_execution": [
        "data_cleaning",
        "log_analysis",
        "unit_test_fix",
        "csv_transformation",
        "regex_extraction",
    ],
    "general_task": [
        "summarization",
        "writing_assistance",
        "decision_support",
        "planning",
        "qa_answering",
    ],
}

DEFAULT_CLI_TOOL_FAMILIES = [
    "filesystem_inspection",
    "file_operations",
    "text_processing",
    "network_fetch",
    "python_runtime",
    "git_workflow",
    "package_management",
    "data_inspection",
    "shell_automation",
    "system_observability",
    "archive_processing",
    "json_yaml_processing",
    "sqlite_inspection",
]


class StrictModel(BaseModel):
    """Base model with explicit schema handling."""

    model_config = ConfigDict(extra="forbid")


class AgentTrajectorySpec(StrictModel):
    """Dataset-specific configuration for agent trajectories."""

    task_types: list[str] = Field(
        default_factory=lambda: ["tool_calling", "code_execution", "general_task"],
        min_length=1,
    )
    task_domains: dict[str, list[str]] = Field(default_factory=lambda: DEFAULT_TASK_DOMAINS.copy())
    cli_tool_families: list[str] = Field(default_factory=lambda: list(DEFAULT_CLI_TOOL_FAMILIES), min_length=1)
    difficulties: list[str] = Field(default_factory=lambda: ["easy", "medium", "hard"], min_length=1)
    step_budgets: list[int] = Field(default_factory=lambda: [4, 5, 6, 7, 8], min_length=1)
    task_type_weights: list[float] | None = None
    cli_tool_family_weights: list[float] | None = None
    difficulty_weights: list[float] | None = None
    step_budget_weights: list[float] | None = None


class SkillQuerySpec(StrictModel):
    """Dataset-specific configuration for skill routing query synthesis."""

    skill_dirs: list[str] = Field(default_factory=list)
    skill_roots: list[str] = Field(default_factory=list)
    queries_per_skill: int = Field(default=12, ge=1)
    query_types: list[str] = Field(
        default_factory=lambda: ["direct", "goal_oriented", "discovery", "ambiguous"],
        min_length=1,
    )
    difficulty_levels: list[str] = Field(default_factory=lambda: ["easy", "medium", "hard"], min_length=1)
    include_negative_samples: bool = True
    negatives_per_query: int = Field(default=2, ge=0)
    max_skill_content_chars: int = Field(default=6000, ge=500)

    @model_validator(mode="after")
    def validate_skill_inputs(self) -> SkillQuerySpec:
        """Require at least one skill source and reject unsupported query types."""

        if not self.skill_dirs and not self.skill_roots:
            raise ValueError("skill_query requires at least one entry in `skill_dirs` or `skill_roots`.")

        allowed_types = {"direct", "goal_oriented", "discovery", "ambiguous"}
        unknown_types = sorted(set(self.query_types) - allowed_types)
        if unknown_types:
            allowed = ", ".join(sorted(allowed_types))
            unknown = ", ".join(unknown_types)
            raise ValueError(f"Unsupported query_types: {unknown}. Allowed values: {allowed}")

        return self


class DatasetConfig(StrictModel):
    """Top-level dataset selection and spec payload."""

    type: str
    spec: dict[str, Any] = Field(default_factory=dict)


class ModelSettings(StrictModel):
    """Model settings used to construct Data Designer configs."""

    provider: str
    alias: str
    name: str
    api_key_env: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1600
    reasoning_effort: str = "medium"
    skip_health_check: bool = True


class RunSettings(StrictModel):
    """Execution settings for preview and generation."""

    output_dir: str = "artifacts"
    dataset_name: str = "dataset"
    num_records: int = Field(default=100, ge=1)
    preview_records: int = Field(default=10, ge=1)


class AppConfig(StrictModel):
    """Full application configuration."""

    dataset: DatasetConfig
    model: ModelSettings
    run: RunSettings
