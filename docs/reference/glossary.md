# Glossary

## Public Terms

| Term | Definition |
| --- | --- |
| `batch` | An IR transform that vectorizes execution over selected input leaves. The public function is `batch(ir, *, in_axes=...)`. |
| `checkpoint` | A primitive that labels an intermediate value with a key and collection. It is transparent unless `collect` or `inject` is active. |
| `collect` | A context manager that captures checkpointed values during IR execution. |
| Collection | A namespace used by checkpoints, `collect`, and `inject` to decide which values belong together. |
| Cotangent | Feedback flowing backward through a pullback. In `autoform`, cotangents are usually text feedback. |
| Custom rule | A rule registered on an `@af.custom` function boundary to override `pushforward`, `pullback`, or `batch` behavior. |
| `dce` | Dead-code elimination, an IR transform that removes equations not needed by selected outputs. |
| `Doc` annotation | A schema description attached with `field @ af.Doc("...")`. |
| Dynamic argument | An input leaf represented by a placeholder during tracing and provided at execution time. |
| Execute | The phase that runs an IR with concrete inputs through `.call(...)` or `.acall(...)`. |
| `fold` | A context manager that evaluates foldable primitive calls immediately during tracing and embeds the result as a literal. |
| `inject` | A context manager that substitutes checkpointed values from a provided dictionary during execution. |
| Instance-first DSL | The schema style where the schema is a value-shaped Python instance, not a separate output class declaration. |
| Intercept | A runtime hook around checkpointed values. `collect` captures intercepted values; `inject` replaces them. |
| IR | The intermediate representation produced by `trace`; it contains input variables, equations, and outputs. |
| `lm_client` | A context manager that changes the active LM client for `lm_call` and `lm_schema_call`. |
| `memoize` | A context manager that caches primitive results within its block. During tracing, it can deduplicate identical primitive calls. |
| `pullback` | An IR transform that propagates output cotangents backward to input cotangents. |
| `pushforward` | An IR transform that propagates input tangents forward to output tangents. |
| Pytree | A nested container/leaf structure that `autoform` can walk. Registered dataclasses can be pytrees. |
| `PYTREE_NAMESPACE` | The optree namespace reserved by `autoform` for user pytree registration. |
| `sched` | An IR transform that groups independent equations for concurrent async execution. |
| Schema | A pytree of schema leaves such as `Str`, `Float`, and `Enum`, used by `lm_schema_call` for structured output. |
| Static argument | An input leaf fixed at trace time by `af.trace(func, static=...)`. The `static` value is a bool pytree matching positional input structure. |
| `Tag` | A hashable metadata object attached to equations during tracing. |
| `tag` | A context manager that activates one or more `Tag` instances for equations created in its block. |
| Trace | The phase that runs a Python function once with placeholders and records `autoform` primitive calls as IR equations. |
| Transform | A function that consumes an IR and returns another IR. The five core IR transforms are `batch`, `pushforward`, `pullback`, `sched`, and `dce`. |

## Internal / IR Machinery

These names are useful when reading internals or debugging a transform. They are
not part of the everyday user surface.

| Term | Definition |
| --- | --- |
| Boxing | The internal technique interpreters use to wrap values with transform-specific metadata. Users normally see the result only through public transforms. |
| `IREqn` | One recorded primitive application in an IR. |
| `IRVar` | A typed placeholder for a runtime value inside an IR. |
| `Prim` | A named primitive operation used as the dispatch key for execution and transform rules. |
| `TraceBox` | The internal wrapper used by the trace interpreter to carry an `IRVar` through Python code. |
| Tracer | The trace-time interpreter machinery that records primitive calls instead of executing them normally. |
