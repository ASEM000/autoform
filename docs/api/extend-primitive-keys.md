# Primitive Keys

Rule registries use primitive instances as dispatch keys. Rules for an existing primitive must refer to the exported instance below. Constructing a new primitive with the same name creates a different key, so the new instance does not inherit the original registrations.

```{eval-rst}
.. autodata:: autoform.string.concat_p
.. autodata:: autoform.string.match_p
.. autodata:: autoform.numeric.neg_p
.. autodata:: autoform.numeric.add_p
.. autodata:: autoform.numeric.sub_p
.. autodata:: autoform.numeric.mul_p
.. autodata:: autoform.numeric.div_p
.. autodata:: autoform.numeric.eq_p
.. autodata:: autoform.numeric.ne_p
.. autodata:: autoform.numeric.lt_p
.. autodata:: autoform.numeric.le_p
.. autodata:: autoform.numeric.gt_p
.. autodata:: autoform.numeric.ge_p
.. autodata:: autoform.lm.fill_p
.. autodata:: autoform.intercept.checkpoint_p
.. autodata:: autoform.control.stop_gradient_p
.. autodata:: autoform.control.switch_p
.. autodata:: autoform.control.while_loop_p
.. autodata:: autoform.control.fixpoint_p
.. autodata:: autoform.path.factor_p
.. autodata:: autoform.path.weight_call_p
.. autodata:: autoform.order.fanout_p
.. autodata:: autoform.order.depends_p
.. autodata:: autoform.axis.batch_call_p
.. autodata:: autoform.ad.pushforward_call_p
.. autodata:: autoform.ad.pullback_call_p
```
