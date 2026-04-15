"""Tests for config loading, registry lookup, and builder assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from openllm_agent_synth.config.loader import ConfigError, load_app_config
from openllm_agent_synth.datasets.base import DatasetBuilderError
from openllm_agent_synth.datasets.registry import get_dataset_builder
from openllm_agent_synth.services.builder_service import build_app_context
from openllm_agent_synth.utils import load_environment

VALID_CONFIG = """
dataset:
  type: agent_trajectory
  spec: {}
model:
  provider: openai
  alias: gpt-5.4-mini
  name: gpt-5.4-mini
  api_key_env: OPENAI_API_KEY
  temperature: 0.7
  top_p: 0.95
  max_tokens: 1600
  reasoning_effort: medium
  skip_health_check: true
run:
  output_dir: artifacts
  dataset_name: agent_trajectory
  num_records: 100
  preview_records: 10
"""


def write_config(tmp_path: Path, content: str = VALID_CONFIG) -> Path:
    """Write a YAML config into the temp directory."""

    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_environment_loads_dotenv(tmp_path, monkeypatch):
    """Load variables from a .env file in the current working directory."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")

    assert load_environment() is True
    assert "OPENAI_API_KEY" in __import__("os").environ


def test_load_environment_does_not_override_existing_values(tmp_path, monkeypatch):
    """Keep already exported variables over .env values."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

    assert load_environment() is True
    assert __import__("os").environ["OPENAI_API_KEY"] == "existing-key"


def test_load_app_config_reads_valid_yaml(tmp_path):
    """Load a complete YAML config into the typed app config."""

    config = load_app_config(write_config(tmp_path))

    assert config.dataset.type == "agent_trajectory"
    assert config.model.name == "gpt-5.4-mini"
    assert config.run.num_records == 100


def test_load_app_config_rejects_missing_required_fields(tmp_path):
    """Fail loudly when mandatory config sections are absent."""

    config_path = write_config(
        tmp_path,
        """
dataset:
  spec: {}
model:
  provider: openai
run:
  output_dir: artifacts
""",
    )

    with pytest.raises(ConfigError):
        load_app_config(config_path)


def test_dataset_registry_returns_agent_trajectory_builder():
    """Resolve the built-in trajectory dataset implementation."""

    builder = get_dataset_builder("agent_trajectory")

    assert builder.dataset_type == "agent_trajectory"


def test_dataset_registry_rejects_unknown_dataset():
    """Reject unsupported dataset types with a clear error."""

    with pytest.raises(DatasetBuilderError, match="Unknown dataset type"):
        get_dataset_builder("does_not_exist")


def test_build_app_context_assembles_expected_columns(tmp_path, monkeypatch):
    """Build a real Data Designer config builder from YAML settings."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = write_config(tmp_path)
    app_config = load_app_config(config_path)

    context = build_app_context(app_config, config_path=config_path)
    column_names = {column.name for column in context.builder.get_column_configs()}

    assert context.builder.model_configs[0].alias == "gpt-5.4-mini"
    assert {"task_type", "task_domain", "cli_tool_family", "task", "trajectory"} <= column_names


def test_build_app_context_uses_api_base_override_from_env(tmp_path, monkeypatch):
    """Allow `.env`/env to override the default provider endpoint."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("openai_api_base", "https://example.test/v1")
    config_path = write_config(tmp_path)
    app_config = load_app_config(config_path)

    context = build_app_context(app_config, config_path=config_path)

    assert context.model_providers[0].endpoint == "https://example.test/v1"


def test_build_app_context_requires_api_key_env(tmp_path, monkeypatch):
    """Validate runtime env requirements before trying to execute Data Designer."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = write_config(tmp_path)
    app_config = load_app_config(config_path)

    with pytest.raises(DatasetBuilderError, match="OPENAI_API_KEY"):
        build_app_context(app_config, config_path=config_path)
