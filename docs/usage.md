# Usage

`openllm-agent-synth` is driven by a single YAML configuration file.

Use the example config at [examples/agent_trajectory.yaml](/Users/ugreen/GitHub/openllm-agent-synth/examples/agent_trajectory.yaml:1)
as a starting point, then run:

```bash
openllm-agent-synth validate -c examples/agent_trajectory.yaml
openllm-agent-synth preview -c examples/agent_trajectory.yaml
openllm-agent-synth generate -c examples/agent_trajectory.yaml
```

The CLI loads `.env` automatically, validates the YAML schema, resolves the built-in
dataset implementation, and then uses the Data Designer Python API to preview or create
artifacts under the configured output directory.
