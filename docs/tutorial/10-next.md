# Next Steps

You have seen the core loop:

- write an LM program as ordinary Python;
- trace it into an IR;
- execute the IR with real inputs;
- transform the IR with `batch`, `pullback`, and `sched`;
- inspect or replace intermediates with `collect` and `inject`;
- return structured values with `lm_schema_call`.

The main conceptual pages are useful once you start building larger programs:

| Need | Go to |
| --- | --- |
| Understand the trace/execution split | [Trace, IR, Execute](../concepts/trace-ir-execute.md) |
| Understand the recorded operations | [Primitives](../concepts/primitives.md) |
| Compose `batch`, `pullback`, and `sched` | [Transforms](../concepts/transforms.md) |
| Inspect or replace intermediate values | [Intercepts](../concepts/intercepts.md) |
| Use structured LM outputs | [Schemas](../concepts/schemas.md) |
| Build a tool-use agent | [Tool-Use Agent](../recipes/tool-use-agent.md) |
| Read API names quickly | [Glossary](../reference/glossary.md) |

For exact call signatures, use the [API Reference](../api.md).

For bugs, design questions, or examples that do not behave as expected, open a [GitHub issue](https://github.com/ASEM000/autoform/issues).
