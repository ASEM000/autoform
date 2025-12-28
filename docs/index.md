# autoform

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
api
```

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

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
