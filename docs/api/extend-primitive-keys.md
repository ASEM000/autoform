# Primitive Keys

Rule registries use primitive instances as dispatch keys. Rules for an existing primitive must refer to the exported instance below. Constructing a new primitive with the same name creates a different key, so the new instance does not inherit the original registrations.

```{eval-rst}
.. autodata:: autoform.extend.p_concat
.. autodata:: autoform.extend.p_match
.. autodata:: autoform.extend.p_neg
.. autodata:: autoform.extend.p_add
.. autodata:: autoform.extend.p_sub
.. autodata:: autoform.extend.p_mul
.. autodata:: autoform.extend.p_div
.. autodata:: autoform.extend.p_eq
.. autodata:: autoform.extend.p_ne
.. autodata:: autoform.extend.p_lt
.. autodata:: autoform.extend.p_le
.. autodata:: autoform.extend.p_gt
.. autodata:: autoform.extend.p_ge
.. autodata:: autoform.extend.p_fill
.. autodata:: autoform.extend.p_checkpoint
.. autodata:: autoform.extend.p_stop_gradient
.. autodata:: autoform.extend.p_switch
.. autodata:: autoform.extend.p_while_loop
.. autodata:: autoform.extend.p_fixpoint
.. autodata:: autoform.extend.p_factor
.. autodata:: autoform.extend.p_weight_call
.. autodata:: autoform.extend.p_fanout
.. autodata:: autoform.extend.p_depends
.. autodata:: autoform.extend.p_batch_call
.. autodata:: autoform.extend.p_pushforward_call
.. autodata:: autoform.extend.p_pullback_call
```
