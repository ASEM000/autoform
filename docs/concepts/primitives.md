# Primitives

Primitives are named operations that are recorded in the [IR](the-ir.md) rather than executed when [tracing](tracing-semantics.md). Examples of primitives include things like {py:func}`concat <autoform.string.concat>`, {py:func}`fill <autoform.lm.fill>`, {py:func}`switch <autoform.switch>`, {py:func}`checkpoint <autoform.checkpoint>`, and {py:func}`factor <autoform.factor>`.

[Transforms](transforms.md) select rules by primitive identity. Two primitives with the same name remain separate rule keys. {py:func}`pullback <autoform.pullback>` knows how to route feedback through the {py:func}`fill <autoform.lm.fill>` primitive because a rule is registered for it. Plain Python operations do not have those rules, so these operations either run at trace time or fail when a concrete runtime value is needed.

## Rule Registries

A primitive can have separate rules for execution, tracing, and each transform. Extension authors register these rules through `autoform.extend`; the internal registries are:

- `impl_rules`: concrete execution.
- `abstract_rules`: output-shape and output-type inference while tracing.
- `batch_rules`: vectorized behavior for {py:func}`batch <autoform.batch>`.
- `push_rules`: forward-mode behavior for {py:func}`pushforward <autoform.pushforward>`.
- `pull_fwd_rules`: the forward sweep for {py:func}`pullback <autoform.pullback>`.
- `pull_bwd_rules`: the backward sweep for {py:func}`pullback <autoform.pullback>`.

The split pullback rules matter: the forward sweep records the values needed later, and the backward sweep uses those residuals plus the cotangent to produce input cotangents.

## Public Primitive Groups

String operations include:

- {py:func}`concat <autoform.string.concat>`: traceable string concatenation.
- {py:func}`match <autoform.string.match>`: traceable string equality.

Note: {py:func}`format <autoform.string.format>` is a helper function to resolve template fields and call `concat`. It is not a primitive, and it does not have rules. Use e.g. `{name}` as field names, to be filled in with keyword arguments. If needed, select attributes or index with python before passing to `format`.

LM generation: {py:func}`fill <autoform.lm.fill>`: replace specs in a pytree with generated values, while retaining context. See [Schemas](schemas.md).

Numeric operations support floating-point arithmetic and scalar comparisons. The registered AD rules compute numerical tangents and cotangents. See [the numeric API](../api/primitives.md#numeric).

control-flow / dependency operations:

- {py:func}`switch <autoform.switch>`: choose one traced branch at execution time.
- {py:func}`while_loop <autoform.while_loop>`: run a traced loop with an explicit iteration cap.
- {py:func}`fixpoint <autoform.fixpoint>`: iterate a traced step function until the state stops changing. See [Fixed Points](fixpoint.md).
- {py:func}`stop_gradient <autoform.stop_gradient>`: pass `x` forward but block cotangents in pullback.
- {py:func}`depends <autoform.depends>`: make a returned result wait for extra dependencies without changing its value.

Intermediate-value inspection uses:

- {py:func}`checkpoint <autoform.checkpoint>`: mark an intermediate value for {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>`.

Path scoring uses:

- {py:func}`factor <autoform.factor>`: multiply the current path weight. Ordinary execution treats it as a no-output effect; {py:func}`weight <autoform.weight>` returns the accumulated path weight.

## Primitive Definitions

A new primitive needs execution and abstract rules. Add batching, AD, or DCE rules for the transforms the operation should support.

Use [Primitive Definitions](../recipes/extending/writing-primitives.md) when an operation cannot run on traced values and must still appear as one IR equation.
