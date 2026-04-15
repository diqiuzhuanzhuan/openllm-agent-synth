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
