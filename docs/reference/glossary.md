# Glossary

## Public Terms

| Term | Definition |
| --- | --- |
| {py:func}`batch <autoform.batch>` | An IR transform that vectorizes execution over selected input leaves. |
| {py:func}`checkpoint <autoform.checkpoint>` | A primitive that labels an intermediate value with a key and collection. It is transparent unless {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>` is active. |
| {py:func}`collect <autoform.collect>` | A context manager that captures checkpointed values during IR execution. |
| Collection | A namespace used by checkpoints, {py:func}`collect <autoform.collect>`, and {py:func}`inject <autoform.inject>` to decide which values belong together. |
| Cotangent | Feedback flowing backward through a pullback. In `autoform`, cotangents are usually text feedback. |
| Custom rule | A rule registered on a {py:func}`custom <autoform.custom>` traceable function boundary to override {py:func}`pushforward <autoform.pushforward>`, {py:func}`pullback <autoform.pullback>`, or {py:func}`batch <autoform.batch>` behavior. |
| {py:func}`dce <autoform.dce>` | Dead-code elimination, an IR transform that removes equations not needed by selected outputs. |
| {py:class}`Doc <autoform.Doc>` annotation | A schema description attached with `field @ af.Doc("...")`. |
| Dynamic argument | An input leaf represented by a placeholder during tracing and provided at execution time. |
| Execute | The phase that runs an IR with concrete inputs through `.call(...)` or `.acall(...)`. |
| {py:func}`factor <autoform.factor>` | A primitive that multiplies the current path weight. It is neutral during ordinary execution and contributes to {py:func}`weighted <autoform.weighted>` results. |
| {py:func}`fixpoint <autoform.fixpoint>` | A higher-order control-flow primitive that repeatedly applies a traced `(State, Theta) -> State` step until the state is stable or `max_iters` is reached. |
| {py:func}`fold <autoform.fold>` | A context manager that evaluates foldable primitive calls immediately during tracing and embeds the result as a literal. |
| {py:func}`inject <autoform.inject>` | A context manager that substitutes checkpointed values from a provided dictionary during execution. |
| Instance-first DSL | The schema style where the schema is a value-shaped Python instance, not a separate output class declaration. |
| Intercept | A runtime hook around checkpointed values. {py:func}`collect <autoform.collect>` captures intercepted values; {py:func}`inject <autoform.inject>` replaces them. |
| IR | The intermediate representation produced by {py:func}`trace <autoform.trace>`; it contains input variables, equations, and outputs. |
| {py:func}`lm_client <autoform.lm_client>` | A context manager that changes the active LM client for {py:func}`lm_call <autoform.lm_call>` and {py:func}`lm_schema_call <autoform.lm_schema_call>`. |
| {py:func}`memoize <autoform.memoize>` | A context manager that caches primitive results within its block. During tracing, it can deduplicate identical primitive calls. |
| {py:func}`pullback <autoform.pullback>` | An IR transform that propagates output cotangents backward to input cotangents. |
| {py:func}`pushforward <autoform.pushforward>` | An IR transform that propagates input tangents forward to output tangents. |
| Pytree | A nested container/leaf structure that `autoform` can walk. Registered dataclasses can be pytrees. |
| {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>` | The optree namespace reserved by `autoform` for user pytree registration. |
| {py:func}`sched <autoform.sched>` | An IR transform that groups independent equations for concurrent async execution. |
| Schema | A pytree of schema leaves such as {py:class}`Str <autoform.Str>`, {py:class}`Float <autoform.Float>`, and {py:class}`Enum <autoform.Enum>`, used by {py:func}`lm_schema_call <autoform.lm_schema_call>` for structured output. |
| Static argument | An input leaf fixed at trace time by {py:func}`trace <autoform.trace>`. The `static` value is a bool pytree matching positional input structure. |
| tag value | A hashable metadata value attached to equations during tracing. |
| {py:func}`tag <autoform.tag>` | A context manager that activates one or more tag values for equations created in its block. |
| Trace | The phase that runs a Python function once with placeholders and records `autoform` primitive calls as IR equations. |
| Transform | A function that consumes an IR and returns another IR. Current IR transforms include {py:func}`batch <autoform.batch>`, {py:func}`pushforward <autoform.pushforward>`, {py:func}`pullback <autoform.pullback>`, {py:func}`sched <autoform.sched>`, {py:func}`dce <autoform.dce>`, and {py:func}`weighted <autoform.weighted>`. |
| {py:func}`weighted <autoform.weighted>` | An IR transform that returns `(output, path_weight)` for one concrete path. |

## Internal IR Machinery

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
| `walk` | The manual IR stepping interface used by execution internals and advanced debugging code. See [Walk](../concepts/walk.md). |
