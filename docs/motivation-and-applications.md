# Motivation and Applications

Programs often need to support more than one task. The same program may be evaluated across examples, inspected at intermediate steps, or transformed to propagate feedback from outputs to inputs. Implementing each task separately repeats knowledge of its operations and structure.

A similar problem arises when writing the forward and backward passes of a neural network by hand. Changing the network can require changes to both passes. Automatic differentiation frameworks avoid this repetition by composing backward rules for individual operations.

`autoform` extends this idea to programs with text, numbers, and user-defined types. Extensions define spaces for values and feedback, along with rules for operations on those values. These rules compose into whole-program transformations. Execution interfaces also allow the same program to be inspected or run under different policies.

For example, a grading program can combine a language model call with arithmetic. Its {py:func}`pullback <autoform.pullback>` returns text feedback for the rubric and a numerical gradient for a points adjustment.

(applications)=
(defining-optimization-methods)=
| Application | Use |
| --- | --- |
| Optimization | An optimization method uses these building blocks in a feedback loop. It supplies an objective, evaluation data, and an update rule. A transformed program evaluates the objective and computes feedback; the update rule uses that feedback to revise the parameters for the next evaluation. Different optimization methods can reuse the same program and transformation rules with different update policies. [Updating Inputs](concepts/transforms.md#updating-inputs) shows a simple update loop. |
| Sensitivity Analysis | {py:func}`pushforward <autoform.pushforward>` and {py:func}`pullback <autoform.pullback>` propagate changes or feedback through a program, even when no input is being optimized. The registered rules determine what the result means. Feedback for a text rubric, for example, can be a structured critique rather than a numerical gradient. |
| Evaluation and Debugging | {py:func}`batch <autoform.batch>` evaluates a program across examples. Execution contexts can capture checkpointed results or substitute controlled values to investigate a failure and test what happens downstream. [Capture and Replacement](concepts/execution.md#capture-and-replacement) shows this workflow. |
| Ranking | {py:func}`weight <autoform.weight>` scores an execution, and {py:func}`batch <autoform.batch>` applies the scoring program to several candidates. The caller supplies the candidates and decides how to compare their scores. [Tool Ranking](recipes/llm/tool-ranking.md) applies this pattern to tool selection. |
| Execution Policies | A custom interpreter can record operation results and tags. A runner can pause at a marked result for human review. A configured model client can manage routing and retries. The application supplies the checks and responses needed for a reliability workflow. [Human Review](recipes/execution/human-review.md) shows one such runner. |
| Adaptive Workflows | Structured state, branches, and bounded loops support programs that choose tools and revise results. [Tool-Use Agent](recipes/llm/tool-use-agent.md) traces such a program, then applies {py:func}`batch <autoform.batch>` and {py:func}`pullback <autoform.pullback>` to it. |
| Efficient Execution | {py:func}`sched <autoform.sched>` overlaps independent calls, {py:func}`dce <autoform.dce>` removes unused computation, and {py:func}`memoize <autoform.memoize>` reuses matching results. A program can also bind fixed context during tracing while leaving task inputs dynamic. [Concurrent Execution](concepts/execution.md#concurrent-execution) and [Static Context](concepts/tracing.md#static-context) demonstrate these cases. |

New domains can extend these applications by registering types, spaces, operations, and the rules needed by each transform. [Array Extension](recipes/extending/array-extension.md) demonstrates this pattern for NumPy arrays. Transformation order and the available rules determine which combinations a program supports.
