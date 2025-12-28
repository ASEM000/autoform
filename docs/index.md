# autoform

Composable transformations for LLM programs.

## Installation

```bash
pip install autoform
```

## Quick Example

```python
import autoform as af

def greet(name):
    return af.concat("Hello, ", name)

ir = af.build_ir(greet)("name")
print(af.call(ir)("World"))  # "Hello, World"
```

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

quickstart
internals
api
```
