# MIT License
#
# Copyright (c) 2026, Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
