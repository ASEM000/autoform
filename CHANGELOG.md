# Changelog

## vnext (unreleased)

### Breaking Changes

  - Removed `af.Struct`, `af.lm_struct_call`, and the Struct-specific type-tree helpers. Use Optree-registered pytrees under {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>` for object structure, and use {py:func}`lm_schema_call <autoform.lm_schema_call>` with `schema=...` for structured LM outputs.

    ```python
    import optree
    import autoform as af


    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        reasoning: str
        answer: float


    schema = Answer(
        reasoning=af.Str() @ af.Doc("Reasoning behind the answer."),
        answer=af.Float() @ af.Doc("Numeric answer."),
    )
    ```

  - Removed the old `split` / `splitpoint` and primitive-local intercept/effect APIs from the public surface.

  - Public tracing and IR execution boundaries are now positional-only. {py:func}`trace <autoform.trace>`, `IR.call`, `IR.acall`, and related APIs reject keyword arguments so input normalization stays consistent across transforms like `static` and `in_axes`. Use the methods on traced IR objects rather than removed top-level `af.call(...)` / `af.acall(...)` helpers.

    ```python
    def label(item, punctuation):
        return af.format("item: {}{}", item, punctuation)


    ir = af.trace(label)("alpha", "!")
    ir.call("beta", "?")
    # "item: beta?"
    ```

  - {py:func}`lm_call <autoform.lm_call>` and {py:func}`lm_schema_call <autoform.lm_schema_call>` now keep only `model=` as the LM-control input. Provider-specific controls such as `temperature`, `max_tokens`, retries, fallbacks, and rate limits should be configured on the active client, for example with a `litellm.Router` model alias and `litellm_params`.

  - Removed `af.Tag`. Pass any hashable value directly to {py:func}`tag <autoform.tag>` instead of subclassing `af.Tag`.

    ```python
    with af.tag("draft"):
        ir = af.trace(program)("seed")
    ```

### New Features

  - {py:func}`trace <autoform.trace>` with `static=...` now accepts a bool pytree over the positional input structure. Static leaves are fixed at trace time, which lets ordinary Python control flow specialize to one path.

    ```python
    def label(is_error, value):
        if is_error:
            return af.format("error: {}", value)
        return af.format("ok: {}", value)


    ir = af.trace(label, static=(True, False))(True, "disk full")
    ir.call(True, "timeout")
    # "error: timeout"
    ```

  - {py:func}`lm_client <autoform.lm_client>` context manager to set the active LM client (for example a configured `litellm.Router`). Enables concurrency limits, retries, fallbacks, and rate limiting. Check [LiteLLM docs](https://docs.litellm.ai/docs/routing) for reference.

    ```python
    import autoform as af
    from litellm import Router

    litellm_params = dict(model="gpt-5.2", tpm=100_000, rpm=1_000)
    model_list = [dict(model_name="gpt-5.2", litellm_params=litellm_params)]
    client = Router(model_list=model_list, max_parallel_requests=10)


    def program(topic):
        prompt = af.format("Summarize {} in one sentence.", topic)
        return af.lm_call([dict(role="user", content=prompt)], model="gpt-5.2")


    ir = af.trace(program)("topic")
    with af.lm_client(client):
        result = ir.call("release notes")
    ```

  - Added {py:func}`lm_schema_call <autoform.lm_schema_call>` and schema nodes ({py:class}`Str <autoform.Str>`, {py:class}`Int <autoform.Int>`, {py:class}`Float <autoform.Float>`, {py:class}`Bool <autoform.Bool>`, {py:class}`Enum <autoform.Enum>`, {py:class}`Doc <autoform.Doc>`) for structured LM outputs. Schemas are ordinary pytrees and can be dictionaries, lists, tuples, or custom Optree-registered objects.

    ```python
    schema = {
        "kind": af.Enum("summary", "definition"),
        "score": af.Float(min=0, max=1),
        "text": af.Str(),
    }

    out = af.lm_schema_call(messages, model="gpt-5.2", schema=schema)
    ```

### Improvements

  - {py:func}`trace <autoform.trace>` now treats `int`, `float`, and `bool` input leaves as dynamic inputs instead of silently baking them in as literals. Unsupported input leaves now fail fast at trace time instead of being treated as constants.

  - {py:func}`concat <autoform.concat>` and {py:func}`match <autoform.match>` now validate input types during tracing. Ill-typed programs fail during abstract evaluation instead of building invalid IR and crashing later at execution.

    ```python
    def bad(x, y, z):
        return af.concat(x, y, z)


    af.trace(bad)("a", "b", 1)  # AssertionError during tracing
    ```

  - {py:func}`batch <autoform.batch>` now preserves its batch axis at the HOP boundary. if an inner batch rule returns a scalar leaf, the HOP broadcasts it back into the common batch container instead of dropping the axis on that output.

    ```python
    def program(x, y):
        return af.format("x={}", x), af.format("y={}", y)


    ir = af.trace(program)("...", "...")
    batched = af.batch(ir, in_axes=(True, False))

    batched.call(["a", "b"], "constant")
    # (["x=a", "x=b"], ["y=constant", "y=constant"])
    ```

  - {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` documentation now distinguishes execution-time checkpoint collection/substitution from trace-time specialization. {py:func}`collect <autoform.collect>` is documented as execution-only; {py:func}`inject <autoform.inject>` is documented as runtime checkpoint substitution around `ir.call(...)` and trace-time checkpoint specialization when used inside the function being traced.

  - `autoform.analysis.ir_tree_used_ir_vars` is now part of the module's documented export surface because {py:func}`dce <autoform.dce>` depends on it for partial-output liveness.

### Documentation

  - Added a published [Changelog](docs/reference/changelog.md) page under Reference.

  - Reworked [Getting Started](docs/getting-started.md), [Concepts](docs/concepts/index.md), [Recipes](docs/recipes/index.md), and [API Reference](docs/api/index.md) around the trace/transform/execute model, transform composition, schemas, pytrees, custom rules, and primitive authoring.

  - Normalized Mermaid diagrams across the README and docs, and added a GitHub footer link to the Furo docs theme.

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

  - `sched` and `depends` for concurrent execution with explicit dependency tracking
  
    ```python
    def program(x):
        a = af.format("[{}]", x)
        b = af.format("<{}>", x)
        return a, b


    ir = af.trace(program)("...")
    scheduled = af.sched(ir)
    result = await scheduled.acall("A")  # ("[A]", "<A>")
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

  - Language model integration via `lm_call` and `lm_struct_call` (powered by LiteLLM)
  
    ```python
    def explain(topic):
        msg = dict(role="user", content=af.format("Explain {}", topic))
        return af.lm_call([msg], model="gpt-5.2")
    ```
