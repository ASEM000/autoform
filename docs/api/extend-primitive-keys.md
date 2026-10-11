# Primitive Keys

Rule registries use primitive instances as dispatch keys. Rules for an existing primitive must refer to the exported instance below. Constructing a new primitive with the same name creates a different key, so the new instance does not inherit the original registrations.

```{eval-rst}
.. autodata:: autoform.extend.concat_p
.. autodata:: autoform.extend.match_p
.. autodata:: autoform.extend.neg_p
.. autodata:: autoform.extend.add_p
.. autodata:: autoform.extend.sub_p
.. autodata:: autoform.extend.mul_p
.. autodata:: autoform.extend.div_p
.. autodata:: autoform.extend.eq_p
.. autodata:: autoform.extend.ne_p
.. autodata:: autoform.extend.lt_p
.. autodata:: autoform.extend.le_p
.. autodata:: autoform.extend.gt_p
.. autodata:: autoform.extend.ge_p
.. autodata:: autoform.extend.fill_p
.. autodata:: autoform.extend.checkpoint_p
.. autodata:: autoform.extend.stop_gradient_p
.. autodata:: autoform.extend.switch_p
.. autodata:: autoform.extend.while_loop_p
.. autodata:: autoform.extend.fixpoint_p
.. autodata:: autoform.extend.factor_p
.. autodata:: autoform.extend.weight_call_p
.. autodata:: autoform.extend.fanout_p
.. autodata:: autoform.extend.depends_p
.. autodata:: autoform.extend.batch_call_p
.. autodata:: autoform.extend.pushforward_call_p
.. autodata:: autoform.extend.pullback_call_p
```
