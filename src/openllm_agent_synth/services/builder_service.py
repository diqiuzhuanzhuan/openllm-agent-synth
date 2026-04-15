"""Build runtime objects from validated application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("DATA_DESIGNER_HOME", str(Path.cwd() / ".data-designer-home"))

import data_designer.config as dd

from openllm_agent_synth.config.models import AppConfig
from openllm_agent_synth.datasets.base import DatasetBuilderError
from openllm_agent_synth.datasets.registry import get_dataset_builder

PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

PROVIDER_ENDPOINT_ENV_VARS = {
    "openai": ("OPENAI_API_BASE", "openai_api_base"),
    "nvidia": ("NVIDIA_API_BASE", "nvidia_api_base"),
    "openrouter": ("OPENROUTER_API_BASE", "openrouter_api_base"),
}


@dataclass(frozen=True)
class AppBuildContext:
    """Resolved runtime objects derived from the YAML config."""

    app_config: AppConfig
    config_path: Path
    output_dir: Path
    dataset_builder_name: str
    builder: dd.DataDesignerConfigBuilder
    model_providers: list[dd.ModelProvider]


def build_app_context(app_config: AppConfig, *, config_path: str | Path) -> AppBuildContext:
    """Resolve dataset implementation and runtime settings."""

    dataset_impl = get_dataset_builder(app_config.dataset.type)
    typed_spec = dataset_impl.parse_spec(app_config.dataset.spec)
    builder = dataset_impl.build(typed_spec, app_config.model)
    output_dir = Path(app_config.run.output_dir)

    _validate_runtime_environment(app_config)

    provider_name = app_config.model.provider.lower()
    if provider_name not in PROVIDER_ENDPOINTS:
        supported = ", ".join(sorted(PROVIDER_ENDPOINTS))
        raise DatasetBuilderError(f"Unsupported model provider '{provider_name}'. Supported providers: {supported}")

    model_provider = dd.ModelProvider(
        name=provider_name,
        endpoint=_resolve_provider_endpoint(provider_name),
        provider_type="openai",
        api_key=app_config.model.api_key_env,
    )

    return AppBuildContext(
        app_config=app_config,
        config_path=Path(config_path),
        output_dir=output_dir,
        dataset_builder_name=dataset_impl.dataset_type,
        builder=builder,
        model_providers=[model_provider],
    )


def _validate_runtime_environment(app_config: AppConfig) -> None:
    """Ensure required environment variables are present before execution."""

    env_name = app_config.model.api_key_env
    if not os.environ.get(env_name):
        raise DatasetBuilderError(f"Required environment variable '{env_name}' is not set.")


def _resolve_provider_endpoint(provider_name: str) -> str:
    """Resolve the provider endpoint, allowing env overrides from `.env`."""

    for env_name in PROVIDER_ENDPOINT_ENV_VARS.get(provider_name, ()):
        if value := os.environ.get(env_name):
            return value
    return PROVIDER_ENDPOINTS[provider_name]
