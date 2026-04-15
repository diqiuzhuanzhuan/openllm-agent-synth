"""Execution helpers that run validate, preview, and generate actions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from openllm_agent_synth.config.loader import load_app_config
from openllm_agent_synth.services.builder_service import build_app_context
from openllm_agent_synth.utils import load_environment


@dataclass(frozen=True)
class ValidationSummary:
    """Validation output for CLI display and tests."""

    config_path: Path
    dataset_type: str
    output_dir: Path


@dataclass(frozen=True)
class PreviewSummary:
    """Preview output for CLI display and tests."""

    config_path: Path
    dataset_type: str
    output_dir: Path
    preview_path: Path
    num_records: int


@dataclass(frozen=True)
class GenerationSummary:
    """Generation output for CLI display and tests."""

    config_path: Path
    dataset_type: str
    output_dir: Path
    dataset_path: Path
    num_records: int


def validate_config(config_path: str | Path) -> ValidationSummary:
    """Validate YAML config and compiled Data Designer builder."""

    context = _load_context(config_path)
    data_designer = _create_data_designer(context.output_dir, context.model_providers)
    data_designer.validate(context.builder)
    return ValidationSummary(
        config_path=context.config_path,
        dataset_type=context.dataset_builder_name,
        output_dir=context.output_dir,
    )


def preview_dataset(config_path: str | Path) -> PreviewSummary:
    """Run preview and persist the preview dataset as JSON."""

    context = _load_context(config_path)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    data_designer = _create_data_designer(context.output_dir, context.model_providers)
    preview_results = data_designer.preview(
        context.builder,
        num_records=context.app_config.run.preview_records,
    )
    preview_path = context.output_dir / f"{context.app_config.run.dataset_name}_preview.json"
    records = []
    if preview_results.dataset is not None:
        records = preview_results.dataset.to_dict(orient="records")
    preview_path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
    return PreviewSummary(
        config_path=context.config_path,
        dataset_type=context.dataset_builder_name,
        output_dir=context.output_dir,
        preview_path=preview_path,
        num_records=context.app_config.run.preview_records,
    )


def generate_dataset(config_path: str | Path) -> GenerationSummary:
    """Run full dataset generation and report the generated artifact path."""

    context = _load_context(config_path)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    data_designer = _create_data_designer(context.output_dir, context.model_providers)
    results = data_designer.create(
        context.builder,
        num_records=context.app_config.run.num_records,
        dataset_name=context.app_config.run.dataset_name,
    )
    return GenerationSummary(
        config_path=context.config_path,
        dataset_type=context.dataset_builder_name,
        output_dir=context.output_dir,
        dataset_path=results.artifact_storage.base_dataset_path,
        num_records=context.app_config.run.num_records,
    )


def _load_context(config_path: str | Path):
    """Load `.env`, parse YAML config, and resolve runtime objects."""

    load_environment()
    app_config = load_app_config(config_path)
    return build_app_context(app_config, config_path=config_path)


def _create_data_designer(output_dir: Path, model_providers):
    """Create a Data Designer instance with an isolated writable home directory."""

    os.environ.setdefault("DATA_DESIGNER_HOME", str(output_dir / ".data-designer-home"))
    from data_designer.interface import DataDesigner

    return DataDesigner(
        artifact_path=output_dir,
        model_providers=model_providers,
    )
