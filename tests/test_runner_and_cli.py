"""Tests for runner integration and CLI command behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner

from openllm_agent_synth.cli import app
from openllm_agent_synth.config.loader import ConfigError
from openllm_agent_synth.services import runner as runner_module
from openllm_agent_synth.services.runner import GenerationSummary, PreviewSummary, ValidationSummary

runner = CliRunner()


@dataclass
class FakeAppConfig:
    run: object


@dataclass
class FakeRunSettings:
    output_dir: str
    dataset_name: str
    num_records: int
    preview_records: int


@dataclass
class FakeContext:
    app_config: FakeAppConfig
    config_path: Path
    output_dir: Path
    dataset_builder_name: str
    builder: object
    model_providers: list[object]


class FakePreviewDataset:
    """Mimic the DataFrame interface used by preview serialization."""

    def to_dict(self, orient: str):
        assert orient == "records"
        return [{"task": "sample"}]


class FakePreviewResults:
    dataset = FakePreviewDataset()


class FakeArtifactStorage:
    def __init__(self, base_dataset_path: Path):
        self.base_dataset_path = base_dataset_path


class FakeGenerationResults:
    def __init__(self, base_dataset_path: Path):
        self.artifact_storage = FakeArtifactStorage(base_dataset_path)


class FakeDataDesigner:
    """Capture runner calls without hitting the real interface layer."""

    def __init__(self, artifact_path, model_providers):
        self.artifact_path = Path(artifact_path)
        self.model_providers = model_providers
        self.validate_calls = []
        self.preview_calls = []
        self.create_calls = []

    def validate(self, builder):
        self.validate_calls.append(builder)

    def preview(self, builder, *, num_records):
        self.preview_calls.append((builder, num_records))
        return FakePreviewResults()

    def create(self, builder, *, num_records, dataset_name):
        self.create_calls.append((builder, num_records, dataset_name))
        return FakeGenerationResults(self.artifact_path / dataset_name)


def make_context(tmp_path: Path) -> FakeContext:
    """Build a minimal runner context for tests."""

    return FakeContext(
        app_config=FakeAppConfig(run=FakeRunSettings("artifacts", "agent_trajectory", 100, 10)),
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "artifacts",
        dataset_builder_name="agent_trajectory",
        builder=object(),
        model_providers=[object()],
    )


def test_validate_config_returns_summary(tmp_path, monkeypatch):
    """Runner validate should compile the builder and return a summary."""

    context = make_context(tmp_path)
    fake_data_designer = FakeDataDesigner(context.output_dir, context.model_providers)
    monkeypatch.setattr(runner_module, "_load_context", lambda _: context)
    monkeypatch.setattr(runner_module, "_create_data_designer", lambda output_dir, model_providers: fake_data_designer)

    summary = runner_module.validate_config(context.config_path)

    assert isinstance(summary, ValidationSummary)
    assert summary.dataset_type == "agent_trajectory"
    assert fake_data_designer.validate_calls == [context.builder]


def test_preview_dataset_writes_json_file(tmp_path, monkeypatch):
    """Runner preview should persist preview rows and pass preview count through."""

    context = make_context(tmp_path)
    fake_data_designer = FakeDataDesigner(context.output_dir, context.model_providers)
    monkeypatch.setattr(runner_module, "_load_context", lambda _: context)
    monkeypatch.setattr(runner_module, "_create_data_designer", lambda output_dir, model_providers: fake_data_designer)

    summary = runner_module.preview_dataset(context.config_path)

    assert isinstance(summary, PreviewSummary)
    assert summary.preview_path.exists()
    assert fake_data_designer.preview_calls == [(context.builder, 10)]


def test_generate_dataset_returns_dataset_path(tmp_path, monkeypatch):
    """Runner generate should pass record count and dataset name through."""

    context = make_context(tmp_path)
    fake_data_designer = FakeDataDesigner(context.output_dir, context.model_providers)
    monkeypatch.setattr(runner_module, "_load_context", lambda _: context)
    monkeypatch.setattr(runner_module, "_create_data_designer", lambda output_dir, model_providers: fake_data_designer)

    summary = runner_module.generate_dataset(context.config_path)

    assert isinstance(summary, GenerationSummary)
    assert summary.dataset_path == context.output_dir / "agent_trajectory"
    assert fake_data_designer.create_calls == [(context.builder, 100, "agent_trajectory")]


def test_cli_validate_success(monkeypatch, tmp_path):
    """CLI validate should show dataset type and output dir."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "openllm_agent_synth.cli.validate_config",
        lambda path: ValidationSummary(
            config_path=Path(path),
            dataset_type="agent_trajectory",
            output_dir=tmp_path / "artifacts",
        ),
    )

    result = runner.invoke(app, ["validate", "-c", str(config_path)])

    assert result.exit_code == 0
    assert "Validated dataset `agent_trajectory`" in result.stdout


def test_cli_validate_failure(monkeypatch, tmp_path):
    """CLI validate should render config errors with exit code 1."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: {}\n", encoding="utf-8")

    def raise_error(path):
        raise ConfigError("broken config")

    monkeypatch.setattr("openllm_agent_synth.cli.validate_config", raise_error)

    result = runner.invoke(app, ["validate", "-c", str(config_path)])

    assert result.exit_code == 1
    assert "broken config" in result.stdout


def test_cli_preview_success(monkeypatch, tmp_path):
    """CLI preview should print the saved preview path."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: {}\n", encoding="utf-8")
    preview_path = tmp_path / "artifacts" / "preview.json"
    monkeypatch.setattr(
        "openllm_agent_synth.cli.preview_dataset",
        lambda path: PreviewSummary(
            config_path=Path(path),
            dataset_type="agent_trajectory",
            output_dir=tmp_path / "artifacts",
            preview_path=preview_path,
            num_records=10,
        ),
    )

    result = runner.invoke(app, ["preview", "-c", str(config_path)])

    assert result.exit_code == 0
    assert str(preview_path) in result.stdout


def test_cli_generate_success(monkeypatch, tmp_path):
    """CLI generate should print the dataset artifact path."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: {}\n", encoding="utf-8")
    dataset_path = tmp_path / "artifacts" / "agent_trajectory"
    monkeypatch.setattr(
        "openllm_agent_synth.cli.generate_dataset",
        lambda path: GenerationSummary(
            config_path=Path(path),
            dataset_type="agent_trajectory",
            output_dir=tmp_path / "artifacts",
            dataset_path=dataset_path,
            num_records=100,
        ),
    )

    result = runner.invoke(app, ["generate", "-c", str(config_path)])

    assert result.exit_code == 0
    assert str(dataset_path) in result.stdout
