<div align="center">

# `autoform`

**Trace once. Transform freely.**

Composable function transformations for LM programs.

*Think [JAX](https://github.com/jax-ml/jax), but for LM programs.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ASEM000/autoform/actions/workflows/ci.yml/badge.svg)](https://github.com/ASEM000/autoform/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ASEM000/autoform/graph/badge.svg?token=Z0JBHSC3ZK)](https://codecov.io/gh/ASEM000/autoform)
[![Documentation](https://readthedocs.org/projects/autoform/badge/?version=latest)](https://autoform.readthedocs.io)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

[Quickstart](#quickstart) · [Transforms](#transforms) · [Concurrency](#concurrency) · [Debugging](#debugging) · [Docs](https://autoform.readthedocs.io)

</div>

```bash
pip install git+https://github.com/ASEM000/autoform.git
```

## Quickstart
```python
import autoform as af

def explain(topic: str) -> str:
    prompt = af.format("Explain {} in one paragraph.", topic)
    msg = dict(role="user", content=prompt)
    return af.lm_call([msg], model="gpt-4o")

ir = af.trace(explain)("...")  # capture structure, no execution
```

Now transform it:
```python
# execute
output = ir.call("quantum entanglement")

# batch: n inputs
outputs = af.batch(ir).call(["DNA", "gravity", "recursion"])

# differentiate: critique output and get input prompt improvement hint  
output, hint = af.pullback(ir).call(("quantum entanglement", "too technical"))

# compose: batched differentiation
topics = ["DNA", "gravity", "recursion"]
critiques = ["too technical", "too brief", "too abstract"]
outputs, hints = af.batch(af.pullback(ir)).call((topics, critiques))
```

The last line is the point: `batch(pullback(ir))`, transformations compose.

## Transforms

| Transform | What it does |
|-----------|--------------|
| `trace` | Capture program as IR |
| `call` / `acall` | Execute (sync / async) |
| `batch` | Vectorize over inputs |
| `pullback` | Backprop feedback |
| `collect` / `inject` | Checkpoint and replay |
| `sched` | Auto-concurrent execution |
| `dce` | Optimize IR |

## Concurrency

`sched` finds independent LM calls. `acall` runs them concurrently.
```python
scheduled = af.sched(ir)
result = await scheduled.acall("input") # acall for async
```

## Debugging

Checkpoint intermediate values. Replay with modifications.
```python
def pipeline(x: str) -> str:
    msg1 = dict(role="user", content=x)
    step1 = af.lm_call([msg1], model="gpt-4o")
    step1 = af.checkpoint(step1, key="step1", collection="debug")
    
    msg2 = dict(role="user", content=step1)
    step2 = af.lm_call([msg2], model="gpt-4o")
    return step2

ir = af.trace(pipeline)("...")

# capture
with af.collect(collection="debug") as captured:
    result = ir.call("input")

# replay with different step1
with af.inject(collection="debug", values=dict(step1=["modified"])):
    result = ir.call("input")
```

---

> ⚠️ **Early development**: API may change.