# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Loong Ma

"""CLI for YAML-driven dataset synthesis workflows."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from openllm_agent_synth.config.loader import ConfigError
from openllm_agent_synth.datasets.base import DatasetBuilderError
from openllm_agent_synth.services.runner import (
    generate_dataset,
    preview_dataset,
    validate_config,
)

app = typer.Typer(no_args_is_help=True)
console = Console()
CONFIG_OPTION = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True)


def _handle_error(exc: Exception) -> None:
    """Render a consistent CLI error and exit non-zero."""

    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(code=1) from exc


@app.command("validate")
def validate(config: Path = CONFIG_OPTION) -> None:
    """Validate a config file and compiled dataset builder."""

    try:
        summary = validate_config(config)
    except (ConfigError, DatasetBuilderError, ValueError) as exc:
        _handle_error(exc)
    console.print(
        f"Validated dataset `{summary.dataset_type}` from {summary.config_path} with output dir {summary.output_dir}."
    )


@app.command("preview")
def preview(config: Path = CONFIG_OPTION) -> None:
    """Preview a dataset and save preview rows as JSON."""

    try:
        summary = preview_dataset(config)
    except (ConfigError, DatasetBuilderError, ValueError, RuntimeError) as exc:
        _handle_error(exc)
    console.print(f"Previewed {summary.num_records} records for `{summary.dataset_type}`.")
    console.print(f"Preview JSON saved to: {summary.preview_path}", soft_wrap=True)


@app.command("generate")
def generate(config: Path = CONFIG_OPTION) -> None:
    """Generate a dataset and save artifacts locally."""

    try:
        summary = generate_dataset(config)
    except (ConfigError, DatasetBuilderError, ValueError, RuntimeError) as exc:
        _handle_error(exc)
    console.print(f"Generated {summary.num_records} records for `{summary.dataset_type}`.")
    console.print(f"Dataset artifacts saved to: {summary.dataset_path}", soft_wrap=True)


if __name__ == "__main__":
    app()
