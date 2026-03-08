# Changelog

## vnext (unreleased)

### Breaking Changes

  - `Struct` fields are now restricted to leaf types (`str`, `int`, `float`, `bool`), `Struct` subclasses, `Literal[v, ...]`, and fixed-size containers (`Annotated[list[T], Len(n, n)]`). Previously dyamic containers (e.g., `list`, `dict`) and unions were allowed, but they complicate static analysis and tracing. Use nested `Struct`s for complex data.

  - `Struct` instances are now **frozen** (immutable). Direct attribute assignment after construction raises an error. Use `model_construct()` or pytree `unflatten` for bypass semantics (e.g., tracing, batch axes).

  - `trace` now accepts only sync functions. `atrace` has been removed. Async code must be synchronized or wrapped before tracing.

### New Features

  - `using_router` context manager to set `litellm.Router`. Enables concurrency limits, retries, fallbacks, and rate limiting. Check [LiteLLM docs](https://docs.litellm.ai/docs/routing) for reference.

    ```python
    import autoform as af
    from litellm import Router
    model_list = [dict(model_name="gpt-5.2", litellm_params=dict(model="gpt-5.2", tpm=100_000, rpm=1_000))]
    router = Router(
        model_list=model_list,
        max_parallel_requests=10,
    )
    with af.using_router(router):
        result = af.call(ir)(inputs)
    ```


### Improvements

  - `trace` now treats `int`, `float`, and `bool` input leaves as dynamic inputs instead of silently baking them in as literals. Unsupported input leaves now fail fast at trace time instead of being treated as constants.

  - `concat` and `match` now validate input types during tracing. Ill-typed programs fail during abstract evaluation instead of building invalid IR and crashing later at execution.

    ```python
    def bad(x, y, z):
        return af.concat(x, y, z)

    af.trace(bad)("a", "b", 1)  # AssertionError during tracing
    ```

  - `split` now returns the value marked by `splitpoint`, even when unrelated equations appear before the splitpoint. previously `lhs` could incorrectly return the output of the last preceding equation instead of the marked value.

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
