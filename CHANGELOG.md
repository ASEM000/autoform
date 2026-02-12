# Changelog

## vnext (unreleased)

### Breaking Changes

  - `Struct` fields are now restricted to leaf types (`str`, `int`, `float`, `bool`), `Struct` subclasses, `Literal[v, ...]`, and fixed-size containers (`Annotated[list[T], Len(n, n)]`). Previously dyamic containers (e.g., `list`, `dict`) and unions were allowed, but they complicate static analysis and tracing. Use nested `Struct`s for complex data.

  - `Struct` instances are now **frozen** (immutable). Direct attribute assignment after construction raises an error. Use `model_construct()` or pytree `unflatten` for bypass semantics (e.g., tracing, batch axes).

### Improvements

  - `check_struct_field_type` raises descriptive `TypeError` messages at class definition time, pinpointing the exact field and why its type is invalid.

  - `struct_type_tree(cls)` builds a cached pytree with types as leaves, used by `eval_struct_lm_call` to produce `Var` trees in a single `treelib.map` call instead of manual recursion.

## v0.2.0 (February 7, 2026)

### New Features

  - Core tracing engine (`trace`, `atrace`, `call`, `acall`) for capturing and executing computation graphs
  
    ```python
    import autoform as af
    def greet(name):
        return af.format("Hello, {}!", name)
    ir = af.trace(greet)("...")
    result = af.call(ir)("World")  # "Hello, World!"
    ```

  - Forward-mode-like (`pushforward`) and reverse-mode-like (`pullback`) automatic differentiation
  
    ```python
    ir = af.trace(lambda x: af.format("[{}]", x))("x")
    pf_ir = af.pushforward(ir)
    primal, tangent = af.call(pf_ir)(("input", "perturbation"))
    # primal: "[input]", tangent: "[perturbation]"
    ```

    ```python
    ir = af.trace(lambda x: af.format("<{}>", x))("x")
    pb_ir = af.pullback(ir)
    output, grad = af.call(pb_ir)(("primal", "feedback"))
    # output: "<primal>", grad: "feedback"
    ```

  - `batch` transformation for vectorizing over multiple inputs
  
    ```python
    def shout(text):
        return af.format("{}!", text)
    ir = af.trace(shout)("...")
    batched_ir = af.batch(ir)
    result = af.call(batched_ir)(["hello", "world"])  # ["hello!", "world!"]
    ```

  - `sched` transformation for auto-scheduling concurrent execution of independent operations

    ```python
    def program(x):
        a = af.format("[{}]", x)
        b = af.format("<{}>", x)
        return af.concat(a, b)
    ir = af.trace(program)("x")
    scheduled = af.sched(ir)
    result = await af.acall(scheduled)("test")  # concurrent execution
    ```

  - `memoize` transformation for caching repeated primitive calls

    Runtime deduplication:

    ```python
    def program(x):
        a = af.concat(x, "!")
        b = af.concat(x, "!")  # duplicate call
        return af.concat(a, b)
    ir = af.trace(program)("test")
    with af.memoize():
        result = af.call(ir)("hello")  # caches identical calls at runtime
    ```

    Compile-time deduplication (inside `trace`):

    ```python
    def program(x):
        with af.memoize():
            a = af.concat(x, "!")
            b = af.concat(x, "!")  # deduplicated during tracing
            return a, b
    ir = af.trace(program)("test")
    print(len(ir.ireqns))  # 1 (second call eliminated at trace time)
    ```

  - `dce` transformation for dead code elimination
  
    ```python
    def program(x):
        dead = af.concat(x, "dead")  # unused
        live = af.concat(x, "live")
        return live
    ir = af.trace(program)("x")
    dce_ir = af.dce(ir)  # removes dead code
    ```

  - `checkpoint`, `collect`, and `inject` for tagging and substituting intermediate values
  
    ```python
    def func(x):
        return af.checkpoint(x, key="val", collection="debug")
    ir = af.trace(func)("...")
    with af.collect(collection="debug") as captured:
        result = af.call(ir)("hello")
    # captured == {"val": ["hello"]}
    ```

  - `split` and `splitpoint` for splitting traced programs at marked points
  
    ```python
    def program(x):
        y = af.format("Hello {}", x)
        z = af.splitpoint(y, key="mid")
        return af.format("Result: {}", z)
    ir = af.trace(program)("...")
    lhs, rhs = af.split(ir, key="mid")
    ```

  - `gather` and `depends` for concurrent execution with explicit dependency tracking
  
    ```python
    ir1 = af.trace(lambda x: af.format("[{}]", x))("...")
    ir2 = af.trace(lambda x: af.format("<{}>", x))("...")
    result = af.gather([(ir1, "A"), (ir2, "B")])  # ["[A]", "<B>"]
    ```

  - Custom effects system with default handler support
  
    ```python
    from autoform.core import Effect, EffectInterpreter, using_interpreter
    class MyEffect(Effect): ...
    def handler(prim, effect, in_tree, /, **params):
        out_tree = yield in_tree
        return out_tree + " modified"
    with using_interpreter(EffectInterpreter((MyEffect, handler))):
        result = af.call(ir)("input")
    ```

  - Control flow primitives: `switch`, `while_loop`, `stop_gradient`
  
    ```python
    branches = dict(
        a = af.trace(lambda x: af.concat("A:", x))("..."),
        b = af.trace(lambda x: af.concat("B:", x))("..."),
    )
    result = af.switch("a", branches, "test")  # "A:test"
    ```

  - String primitives: `format`, `concat`, `match`
  
    ```python
    af.format("Hello, {}!", "World")       # "Hello, World!"
    af.format("{a}-{b}", a="x", b="y")     # "x-y"
    af.concat("Hello", " World")           # "Hello World"
    af.match("yes", "yes")                 # True
    ```

  - Language model integration via `lm_call` and `struct_lm_call` (powered by LiteLLM)
  
    ```python
    def explain(topic):
        msg = dict(role="user", content=af.format("Explain {}", topic))
        return af.lm_call([msg], model="gpt-5.2")
    ```
