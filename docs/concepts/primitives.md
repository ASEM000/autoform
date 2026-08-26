# Primitives

A primitive is a named operation that the [IR](the-ir.md) records instead of executing inline during [tracing](tracing-semantics.md). Examples include {py:func}`format <autoform.format>`, {py:func}`concat <autoform.concat>`, {py:func}`lm_call <autoform.lm_call>`, {py:func}`switch <autoform.switch>`, {py:func}`checkpoint <autoform.checkpoint>`, and {py:func}`factor <autoform.factor>`.

The name matters because [transforms](transforms.md) dispatch on primitive identity. {py:func}`pullback <autoform.pullback>` knows how to route feedback through the {py:func}`lm_call <autoform.lm_call>` primitive because a rule is registered for it. Plain Python operations do not have those rules, so they either run at trace time or fail when they need a concrete runtime value.

## Rule Registries

Every primitive can have rules for different phases and transforms:

- `impl_rules`: synchronous execution.
- `abstract_rules`: output-shape and output-type inference while tracing.
- `batch_rules`: vectorized behavior for {py:func}`batch <autoform.batch>`.
- `push_rules`: forward-mode behavior for {py:func}`pushforward <autoform.pushforward>`.
- `pull_fwd_rules`: the forward sweep for {py:func}`pullback <autoform.pullback>`.
- `pull_bwd_rules`: the backward sweep for {py:func}`pullback <autoform.pullback>`.

The split pullback rules matter: the forward sweep records the values needed later, and the backward sweep uses those residuals plus the cotangent to produce input cotangents.

## Public Primitive Groups

**String**

- {py:func}`format <autoform.format>`: traceable string formatting.
- {py:func}`concat <autoform.concat>`: traceable string concatenation.
- {py:func}`match <autoform.match>`: traceable string equality.

**LM**

- {py:func}`lm_call <autoform.lm_call>`: chat completion through the active LM client.
- {py:func}`lm_schema_call <autoform.lm_schema_call>`: structured completion parsed into the [schema](schemas.md) shape.

**Control Flow**

- {py:func}`switch <autoform.switch>`: choose one traced branch at execution time.
- {py:func}`while_loop <autoform.while_loop>`: run a traced loop with an explicit iteration cap.
- {py:func}`fixpoint <autoform.fixpoint>`: iterate a traced step function until the state stops changing. See [Fixed Points](fixpoint.md).
- {py:func}`stop_gradient <autoform.stop_gradient>`: pass `x` forward but block cotangents in pullback.
- {py:func}`depends <autoform.depends>`: make a returned result wait for extra dependencies without changing its value.

**Intercepts**

- {py:func}`checkpoint <autoform.checkpoint>`: mark an intermediate value for {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>`.

**Trace Weight**

- {py:func}`factor <autoform.factor>`: multiply the current path weight. Ordinary execution treats it as a no-output effect; {py:func}`weight <autoform.weight>` returns the accumulated path weight.

## Primitive Definitions

Defining a primitive means defining its behavior under execution, tracing, batching, pushforward, pullback, and sometimes DCE. Most user code should not do that.

Use [Write a Primitive](../recipes/extending/writing-primitives.md) when an operation cannot run on traced values and must still appear as one IR equation.
