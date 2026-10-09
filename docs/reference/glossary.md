# Glossary

## Public Terms

These terms describe tracing, transformations, and execution:

| Term | Definition |
| --- | --- |
| {py:func}`batch <autoform.batch>` | IR transform to vectorize over a selection of input leaves. |
| {py:func}`checkpoint <autoform.checkpoint>` | A primitive that labels an intermediate value with a key and collection. It is transparent unless {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>` is active. |
| {py:func}`collect <autoform.collect>` | Context manager to capture checkpointed values during IR execution. |
| Collection | A namespace used by checkpoints, {py:func}`collect <autoform.collect>`, and {py:func}`inject <autoform.inject>` to decide which values belong together. |
| Cotangent | Feedback flowing backward through a pullback. The registered type determines its meaning, such as text feedback for strings or numerical gradients for floats. |
| Custom rule | A rule to be applied to the {py:func}`pushforward <autoform.pushforward>`, {py:func}`pullback <autoform.pullback>`, or {py:func}`batch <autoform.batch>` of a function boundary marked with {py:func}`custom <autoform.custom>`. |
| {py:func}`dce <autoform.dce>` | IR transform: dead code elimination, removes equations that are not required to produce selected outputs. |
| Schema description | Generation guidance, can be applied to a spec with `desc=` or `spec @ description`. |
| Dynamic argument | An input leaf that will be replaced with a placeholder at trace time and supplied at execution time. |
| Execute | Phase where the IR is run with concrete inputs by calling `.call(...)` or `.acall(...)`. |
| {py:func}`factor <autoform.factor>` | Primitive: multiply the current path weight by a scalar. Neutral when executing normally, but used to produce {py:func}`weight <autoform.weight>` results. |
| {py:func}`fixpoint <autoform.fixpoint>` | A higher-order control-flow primitive that repeatedly applies a traced `(State, Theta) -> State` step until the state is stable or `max_iters` is reached. |
| {py:func}`fold <autoform.fold>` | Context manager to evaluate calls to foldable primitives immediately during tracing and insert the result as a literal. |
| {py:func}`inject <autoform.inject>` | A context manager that substitutes checkpointed values from a provided dictionary during execution. |
| Schema instance | A container of context and specs. The structure of the container determines the shape of the output when calling the model. |
| Intercept | Runtime hook for intercepting checkpointed values. Used by {py:func}`collect <autoform.collect>` to capture values and {py:func}`inject <autoform.inject>` to replace values. |
| IR | The intermediate representation produced by {py:func}`trace <autoform.trace>`; it contains input variables, equations, and outputs. |
| {py:func}`client <autoform.lm.client>` | Context manager to select a client to use for {py:func}`fill <autoform.lm.fill>`. |
| {py:func}`memoize <autoform.memoize>` | A context manager that caches primitive results within its block. During tracing, it can deduplicate identical primitive calls. |
| {py:func}`pullback <autoform.pullback>` | IR transform to propagate cotangents from outputs to inputs. |
| {py:func}`pushforward <autoform.pushforward>` | IR transform to propagate tangents from inputs to outputs. |
| Pytree | A nested container/leaf structure that `autoform` can walk. Registered dataclasses can be pytrees. |
| {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>` | Optree namespace reserved for use by `autoform` when registering user pytrees. |
| {py:func}`sched <autoform.sched>` | IR transform to group equations that can be executed concurrently using async. |
| Schema | A pytree of specifications such as {py:class}`Str <autoform.lm.Str>`, {py:class}`Float <autoform.lm.Float>`, and {py:class}`Enum <autoform.lm.Enum>`, used by {py:func}`fill <autoform.lm.fill>` to describe generated values. |
| Static argument | An input leaf that will be fixed by {py:func}`trace <autoform.trace>`. `static` is a `bool` pytree that should match the structure of the positional inputs to trace. |
| tag value | Hashable metadata that can be attached to equations during tracing. |
| {py:func}`tag <autoform.tag>` | Context manager to apply one or more tags to all equations produced within the block. |
| Trace | The phase that runs a Python function once with placeholders and records `autoform` primitive calls as IR equations. |
| Transform | A function that consumes an IR and returns another IR. Current transforms include: {py:func}`batch <autoform.batch>`, {py:func}`pushforward <autoform.pushforward>`, {py:func}`pullback <autoform.pullback>`, {py:func}`sched <autoform.sched>`, {py:func}`dce <autoform.dce>`, and {py:func}`weight <autoform.weight>`. |
| {py:func}`weight <autoform.weight>` | An IR transform that returns `(output, path_weight)` for one concrete path. |

## Internal IR Machinery

These names may be useful when inspecting internals and debugging transforms, but these names are not part of the normal user surface.

| Term | Definition |
| --- | --- |
| `Eqn` | A single application of a primitive in the IR. |
| `Var` | Typed placeholder for a value that will be supplied when running the IR. |
| `Prim` | A named primitive operation used as the dispatch key for execution and transform rules. |
| `TraceBox` | The internal wrapper used by the trace interpreter to carry a `Var` through Python code. |
| Tracer | The trace-time interpreter machinery that records primitive calls instead of executing the calls normally. |
| `walk` | The manual IR stepping interface used by execution internals and advanced debugging code. [Manual Execution](../concepts/execution.md#manual-execution) describes the stepping interface. |
