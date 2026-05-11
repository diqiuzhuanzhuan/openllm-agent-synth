# openllm-agent-synth

![PyPI version](https://img.shields.io/pypi/v/openllm-agent-synth.svg)
[![Documentation Status](https://readthedocs.org/projects/openllm-agent-synth/badge/?version=latest)](https://openllm-agent-synth.readthedocs.io/en/latest/?version=latest)

A framework for generating datasets for agentic applications.

* PyPI package: https://pypi.org/project/openllm-agent-synth/
* Free software: MIT License
* Documentation: https://openllm-agent-synth.readthedocs.io.

## Configuration

Create a local `.env` file from [.env.example](./.env.example) and set the required API keys:

```dotenv
NVIDIA_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
```

The CLI automatically loads environment variables from a `.env` file in the current working directory.

Dataset generation is configured through a single YAML file. A complete example lives at
[examples/agent_trajectory.yaml](/Users/ugreen/GitHub/openllm-agent-synth/examples/agent_trajectory.yaml:1).
For skill-routing query synthesis, see
[examples/skill_query.yaml](/Users/ugreen/GitHub/openllm-agent-synth/examples/skill_query.yaml:1).

The built-in `skill_query` dataset scans one or more skill directories, reads each
`SKILL.md`, and synthesizes routing examples for that skill. It extracts a compact
profile from the skill name, summary, "use when", and "do not use" guidance, then
generates:

* `query`: a realistic user request that should route to the target skill
* `routing_rationale`: a short explanation of why that skill is the correct match
* `evidence_summary`: a normalized summary of the skill profile used during generation

A minimal `skill_query` config looks like this:

```yaml
dataset:
  type: skill_query
  spec:
    skill_roots:
      - /Users/you/.codex/skills
      - /Users/you/.agents/skills
    queries_per_skill: 12
    query_types:
      - direct
      - goal_oriented
      - discovery
      - ambiguous
    difficulty_levels:
      - easy
      - medium
      - hard
    include_negative_samples: true
    negatives_per_query: 2
```

Notes:

* Set either `skill_dirs` for explicit skill folders or `skill_roots` to recursively scan for `SKILL.md`.
* Supported `query_types` are `direct`, `goal_oriented`, `discovery`, and `ambiguous`.
* `include_negative_samples` and `negatives_per_query` add nearby-but-wrong skills to make routing examples harder.
* At least one skill source is required, and at least two skills are required when negative samples are enabled.

## Usage

Validate a config before running:

```bash
openllm-agent-synth validate -c examples/agent_trajectory.yaml
```

Preview a small sample and save it as JSON under the configured output directory:

```bash
openllm-agent-synth preview -c examples/agent_trajectory.yaml
```

Generate the full dataset artifacts:

```bash
openllm-agent-synth generate -c examples/agent_trajectory.yaml
```

To generate skill-routing queries instead, run the same commands with the skill-query config:

```bash
openllm-agent-synth validate -c examples/skill_query.yaml
openllm-agent-synth preview -c examples/skill_query.yaml
openllm-agent-synth generate -c examples/skill_query.yaml
```
