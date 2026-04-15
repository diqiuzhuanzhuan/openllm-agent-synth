"""Built-in dataset implementation for agent trajectory synthesis."""

from __future__ import annotations

from typing import Any, Literal, cast

import data_designer.config as dd
from pydantic import BaseModel, Field

from openllm_agent_synth.config.models import AgentTrajectorySpec, ModelSettings

from .base import BuiltinDatasetBuilder

DEFAULT_CLI_TOOL_CATALOGS = {
    "filesystem_inspection": "ls, find, rg, tree, stat, pwd, head, tail",
    "text_processing": "cat, sed, awk, cut, sort, uniq, tr, xargs",
    "network_fetch": "curl, wget, jq, httpie, openssl, nslookup, dig",
    "python_runtime": "python, uv, pytest, pip, ipython, python -m json.tool",
    "git_workflow": "git status, git diff, git log, git show, git branch, git checkout",
    "package_management": "uv, pip, npm, npx, cargo, make, docker",
    "data_inspection": "ls, rg, cat, python, jq, git, curl",
}


class TrajectoryStep(BaseModel):
    """One step in a synthetic agent trajectory."""

    step_index: int = Field(ge=1, description="1-based step number within the trajectory.")
    kind: Literal[
        "plan",
        "tool_call",
        "tool_result",
        "code_write",
        "code_run",
        "analysis",
        "final_answer",
    ] = Field(description="The type of action taken at this step.")
    content: str = Field(description="Human-readable description of the step.")
    tool_name: str | None = Field(default=None, description="Tool name used for a tool call step.")
    tool_arguments: dict[str, Any] | None = Field(default=None, description="Arguments passed to the tool.")
    observation: str | None = Field(default=None, description="Observed result after a tool call or code run.")
    code: str | None = Field(default=None, description="Code written or executed at this step.")
    stdout: str | None = Field(default=None, description="Captured standard output from code execution.")
    stderr: str | None = Field(default=None, description="Captured standard error from code execution.")


class AgentTrajectoryRecord(BaseModel):
    """Structured output for one trajectory sample."""

    task_type: Literal["tool_calling", "code_execution", "general_task"] = Field(
        description="Primary agent behavior to emulate."
    )
    task_domain: str = Field(description="Domain or subcategory of the task.")
    cli_tool_family: str = Field(description="The main CLI tool family used in the trajectory.")
    cli_tools_used: list[str] = Field(description="Distinct CLI tools or commands referenced in the trajectory.")
    task: str = Field(description="The user-facing task that the agent should solve.")
    steps: list[TrajectoryStep] = Field(description="Step-by-step synthetic trajectory.")
    final_answer: str = Field(description="The final answer returned by the agent.")
    outcome: Literal["success", "partial", "failure"] = Field(description="Whether the task was completed.")
    notes: str | None = Field(default=None, description="Optional commentary on the trajectory.")


class AgentTrajectoryDatasetBuilder(BuiltinDatasetBuilder):
    """Build a synthetic agent trajectory dataset."""

    dataset_type = "agent_trajectory"
    spec_model = AgentTrajectorySpec

    def build(self, spec: AgentTrajectorySpec, model_settings: ModelSettings) -> dd.DataDesignerConfigBuilder:
        """Create the dataset config builder from dataset and model settings."""

        model_config = dd.ModelConfig(
            alias=model_settings.alias,
            model=model_settings.name,
            provider=model_settings.provider,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                max_tokens=model_settings.max_tokens,
                extra_body={"reasoning_effort": model_settings.reasoning_effort},
            ),
            skip_health_check=model_settings.skip_health_check,
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])

        builder.add_column(
            dd.SamplerColumnConfig(
                name="sample_id",
                sampler_type=dd.SamplerType.UUID,
                params=dd.UUIDSamplerParams(prefix="traj-", short_form=True, uppercase=False),
            )
        )
        builder.add_column(
            dd.SamplerColumnConfig(
                name="task_type",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=cast(list[str | int | float], spec.task_types),
                    weights=spec.task_type_weights,
                ),
            )
        )
        builder.add_column(
            dd.SamplerColumnConfig(
                name="task_domain",
                sampler_type=dd.SamplerType.SUBCATEGORY,
                params=dd.SubcategorySamplerParams(
                    category="task_type",
                    values=cast(dict[str, list[str | int | float]], spec.task_domains),
                ),
            )
        )
        builder.add_column(
            dd.SamplerColumnConfig(
                name="cli_tool_family",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=cast(list[str | int | float], spec.cli_tool_families),
                    weights=spec.cli_tool_family_weights,
                ),
            )
        )
        builder.add_column(
            dd.SamplerColumnConfig(
                name="difficulty",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=cast(list[str | int | float], spec.difficulties),
                    weights=spec.difficulty_weights,
                ),
            )
        )
        builder.add_column(
            dd.SamplerColumnConfig(
                name="step_budget",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=cast(list[str | int | float], spec.step_budgets),
                    weights=spec.step_budget_weights,
                ),
            )
        )
        builder.add_column(
            dd.ExpressionColumnConfig(
                name="trajectory_hint",
                expr=(
                    "{% if task_type == 'tool_calling' %}"
                    "Include at least one tool call and one tool result step. "
                    "The trajectory should show the agent consulting external information before answering."
                    "{% elif task_type == 'code_execution' %}"
                    "Include at least one code_write step and one code_run step. "
                    "The agent should inspect execution output and adjust the solution if needed."
                    "{% else %}"
                    "Avoid tools and code. The trajectory should rely on planning, reasoning, and a final answer."
                    "{% endif %}"
                ),
            )
        )
        builder.add_column(
            dd.ExpressionColumnConfig(
                name="cli_tool_catalog",
                expr=self._build_cli_tool_catalog_expression(spec.cli_tool_families),
            )
        )
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="task",
                model_alias=model_settings.alias,
                system_prompt=(
                    "You write realistic synthetic user tasks for agent training. "
                    "Keep each task self-contained, concrete, and solvable. "
                    "Do not include the solution."
                ),
                prompt=(
                    "Write one realistic user task for a {{ task_type }} agent in the "
                    "{{ task_domain }} domain at {{ difficulty }} difficulty. "
                    "The task should be suitable for a synthetic dataset and should take roughly "
                    "{{ step_budget }} steps to solve. "
                    "The trajectory will use CLI tools from this family: {{ cli_tool_family }}. "
                    "Representative commands: {{ cli_tool_catalog }}. "
                    "Task guidance: {{ trajectory_hint }} "
                    "Return only the user request, with no prefacing commentary."
                ),
                with_trace=dd.TraceType.NONE,
            )
        )
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="trajectory",
                model_alias=model_settings.alias,
                system_prompt=(
                    "You generate synthetic agent trajectories for training data. "
                    "Follow the requested mode faithfully and ensure the trajectory is internally consistent. "
                    "Do not add commentary outside the structured output."
                ),
                prompt=(
                    "Given the following user task:\n"
                    "{{ task }}\n\n"
                    "Generate a synthetic agent trajectory for a {{ task_type }} task in the "
                    "{{ task_domain }} domain at {{ difficulty }} difficulty. "
                    "Target about {{ step_budget }} steps. "
                    "CLI tool family: {{ cli_tool_family }}. "
                    "Representative commands: {{ cli_tool_catalog }}. "
                    "{{ trajectory_hint }}\n\n"
                    "Requirements:\n"
                    "- The steps must be realistic and ordered.\n"
                    "- Tool-calling trajectories should use at least 3 distinct CLI tools "
                    "and include command arguments and observations.\n"
                    "- Code-execution trajectories should include at least 2 CLI tools "
                    "plus code, execution output, and any refinement step.\n"
                    "- General-task trajectories should avoid tool and code steps.\n"
                    "- The tool names in `steps` should come from the CLI tool family "
                    "and should be plausible shell commands.\n"
                    "- The `cli_tools_used` field should summarize the distinct commands used in the trajectory.\n"
                    "- The final answer must resolve the task.\n"
                    "- Keep the output aligned with the schema exactly."
                ),
                output_format=AgentTrajectoryRecord,
                with_trace=dd.TraceType.NONE,
            )
        )
        return builder

    def _build_cli_tool_catalog_expression(self, tool_families: list[str]) -> str:
        """Render a Jinja expression that maps tool families to representative commands."""

        branches: list[str] = []
        for index, family in enumerate(tool_families):
            keyword = "{% if" if index == 0 else "{% elif"
            catalog = DEFAULT_CLI_TOOL_CATALOGS.get(family, DEFAULT_CLI_TOOL_CATALOGS["data_inspection"])
            branches.append(f"{keyword} cli_tool_family == '{family}' %}}{catalog}")
        branches.append("{% else %}ls, rg, cat, python, jq, git, curl{% endif %}")
        return "".join(branches)
