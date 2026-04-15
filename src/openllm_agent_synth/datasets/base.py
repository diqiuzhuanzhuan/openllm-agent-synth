"""Base contract for built-in dataset builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import data_designer.config as dd
from pydantic import BaseModel

from openllm_agent_synth.config.models import ModelSettings


class DatasetBuilderError(ValueError):
    """Raised when a dataset type cannot be resolved or built."""


class BuiltinDatasetBuilder(ABC):
    """Abstract dataset builder used by the registry."""

    dataset_type: str
    spec_model: type[BaseModel]

    def parse_spec(self, raw_spec: dict[str, Any]) -> BaseModel:
        """Validate the dataset-specific spec."""

        return self.spec_model.model_validate(raw_spec or {})

    @abstractmethod
    def build(self, spec: Any, model_settings: ModelSettings) -> dd.DataDesignerConfigBuilder:
        """Create a Data Designer config builder for the selected dataset."""
