# Changelog

## vnext (unreleased)

### Breaking Changes

  - `Struct` fields are now restricted to leaf types (`str`, `int`, `float`, `bool`), `Struct` subclasses, `Literal[v, ...]`, and fixed-size containers (`Annotated[list[T], Len(n, n)]`). Previously dyamic containers (e.g., `list`, `dict`) and unions were allowed, but they complicate static analysis and tracing. Use nested `Struct`s for complex data.

  - `Struct` instances are now **frozen** (immutable). Direct attribute assignment after construction raises an error. Use `model_construct()` or pytree `unflatten` for bypass semantics (e.g., tracing, batch axes).

  - Public tracing and IR execution boundaries are now positional-only. `trace`, `call`, `acall`, and related APIs reject keyword arguments so input normalization stays consistent across transforms like `static` and `in_axes`.

    ```python
    def greet(name, punctuation):
        return af.format("Hello, {}{}", name, punctuation)


    ir = af.trace(greet)("world", "!")
    af.call(ir)("Alice", "?")
    ```

  - The primitive-local observational runtime has been renamed from effect terminology to intercept terminology. Update `Effect` -> `Intercept`, `EffectInterpreter` -> `InterceptorInterpreter`, `using_effect` -> `using_intercept`, `active_effect` -> `active_intercept`, `IREqn.effect` -> `IREqn.intercept`, `effect_p` -> `intercept_p`, `autoform.effects` -> `autoform.intercepts`, and `dce(..., keep_effects=...)` -> `dce(..., keep_intercepts=...)`. The callback passed to `InterceptorInterpreter` is now described as an interceptor rather than a handler.

### New Features

  - `trace(..., static=...)` now accepts a bool pytree over the positional input structure. Static leaves are fixed at trace time, which lets ordinary Python control flow specialize to one path.

    ```python
    def label(is_error, value):
        if is_error:
            return af.format("error: {}", value)
        return af.format("ok: {}", value)


    ir = af.trace(label, static=(True, False))(True, "disk full")
    af.call(ir)(True, "timeout")
    # "error: timeout"
    ```

  - `using_client` context manager to set the active LM client (for example a configured `litellm.Router`). Enables concurrency limits, retries, fallbacks, and rate limiting. Check [LiteLLM docs](https://docs.litellm.ai/docs/routing) for reference.

    ```python
    import autoform as af
    from litellm import Router

    litellm_params = dict(model="gpt-5.2", tpm=100_000, rpm=1_000)
    model_list = [dict(model_name="gpt-5.2", litellm_params=litellm_params)]
    client = Router(model_list=model_list, max_parallel_requests=10)
    with af.using_client(client):
        result = af.call(ir)(inputs)
    ```

  - `lm_call(...)` and `struct_lm_call(...)` now accept first-class `temperature=` and `max_tokens=` inputs. Those scalar LM controls can now vary across calls and batches.

  - Added inference primitives `factor` and `weight`.

    ```python
    def program(x):
        y = af.concat(x, "!")
        af.factor(y, judge=lambda s: float(len(s)))
        return y


    ir = af.trace(program)("x")
    out, total = af.call(af.weight(ir))("ab")
    # ("ab!", 3.0)
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

  - `batch` now preserves its batch axis at the HOP boundary. if an inner batch rule returns a scalar leaf, the HOP broadcasts it back into the common batch container instead of dropping the axis on that output.

    ```python
    def program(x, y):
        return af.format("x={}", x), af.format("y={}", y)


    ir = af.trace(program)("...", "...")
    batched = af.batch(ir, in_axes=(True, False))

    af.call(batched)(["a", "b"], "constant")
    # (["x=a", "x=b"], ["y=constant", "y=constant"])
    ```

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

  - Custom intercept system with default interceptor support
  
    ```python
    from autoform.core import Intercept, InterceptorInterpreter, using_interpreter


    class MyIntercept(Intercept): ...


    def interceptor(prim, intercept, in_tree, /, **params):
        out_tree = yield in_tree
        return out_tree + " modified"


    with using_interpreter(InterceptorInterpreter((MyIntercept, interceptor))):
        result = af.call(ir)("input")
    ```

  - Control flow primitives: `switch`, `while_loop`, `stop_gradient`
  
    ```python
    branches = dict(
        a=af.trace(lambda x: af.concat("A:", x))("..."),
        b=af.trace(lambda x: af.concat("B:", x))("..."),
    )
    result = af.switch("a", branches, "test")  # "A:test"
    ```

  - String primitives: `format`, `concat`, `match`
  
    ```python
    af.format("Hello, {}!", "World")  # "Hello, World!"
    af.format("{a}-{b}", a="x", b="y")  # "x-y"
    af.concat("Hello", " World")  # "Hello World"
    af.match("yes", "yes")  # True
    ```

  - Language model integration via `lm_call` and `struct_lm_call` (powered by LiteLLM)
  
    ```python
    def explain(topic):
        msg = dict(role="user", content=af.format("Explain {}", topic))
        return af.lm_call([msg], model="gpt-5.2")
    ```
