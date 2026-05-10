# Primitives

A primitive is a named operation that the [IR](the-ir.md) records instead of executing inline during [tracing](tracing-semantics.md). Examples include `format`, `concat`, `lm_call`, `switch`, and `checkpoint`.

The name matters because [transforms](transforms.md) dispatch on primitive identity. `pullback` knows how to route feedback through `lm_call` because a rule is registered for the `lm_call` primitive. Plain Python operations do not have those rules, so they either run at trace time or fail when they need a concrete runtime value.

## Rule Registries

Every primitive can have rules for different phases and transforms:

- **`impl_rules`**: synchronous execution.
- **`abstract_rules`**: output-shape and output-type inference while tracing.
- **`batch_rules`**: vectorized behavior for `batch`.
- **`push_rules`**: forward-mode behavior for `pushforward`.
- **`pull_fwd_rules`**: the forward sweep for `pullback`.
- **`pull_bwd_rules`**: the backward sweep for `pullback`.

The split pullback rules matter: the forward sweep records the values needed later, and the backward sweep uses those residuals plus the cotangent to produce input cotangents.

## Public Primitive Groups

**String**

- `format(template: str, *args, **kwargs) -> str`: traceable string formatting.
- `concat(*args) -> str`: traceable string concatenation.
- `match(a: str, b: str, /) -> bool`: traceable string equality.

**LM**

- `lm_call(messages, /, *, model: str) -> str`: chat completion through the active LM client.
- `lm_schema_call(messages, /, *, model: str, schema) -> Any`: structured completion parsed into the [schema](schemas.md) shape.

**Control flow**

- `switch(key: str, branches: dict[str, IR], *args) -> Tree`: choose one traced branch at execution time.
- `while_loop(cond_ir, body_ir, init_val, *, max_iters: int) -> Tree`: run a traced loop with an explicit iteration cap.
- `stop_gradient(x) -> Tree`: pass `x` forward but block cotangents in pullback.
- `depends(value, /, *deps) -> value`: force ordering without changing the returned value.

**Intercepts**

- `checkpoint(value, /, *, key, collection=None) -> Tree`: mark an intermediate value for `collect` or `inject`.

## Why User Code Usually Does Not Define Primitives

Defining a primitive means defining its behavior under execution, tracing, batching, pushforward, pullback, and sometimes DCE. Most user code should not do that.

Use [`@af.custom`](custom-rules.md) when a Python function needs a boundary with transform-specific behavior. That gives the control point without requiring a new primitive from scratch.
