"""Pydantic models for YAML-driven application configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    "text_processing",
    "network_fetch",
    "python_runtime",
    "git_workflow",
    "package_management",
    "data_inspection",
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
