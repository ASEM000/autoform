import autoform as af


class TestStruct:
    def test_struct_is_pytree(self):
        class Answer(af.Struct):
            reasoning: str
            answer: int

        a = Answer(reasoning="think step by step", answer=42)
        leaves = af.utils.treelib.leaves(a)
        assert leaves == ["think step by step", 42]

    def test_struct_unflatten(self):
        class Answer(af.Struct):
            reasoning: str
            answer: int

        a = Answer(reasoning="original", answer=1)
        spec = af.utils.treelib.structure(a)
        restored = spec.unflatten(["new reasoning", 100])
        assert restored.reasoning == "new reasoning"
        assert restored.answer == 100
        assert isinstance(restored, Answer)

    def test_struct_unflatten_skips_validation(self):
        class Positive(af.Struct):
            value: int

        spec = af.utils.treelib.structure(Positive(value=1))
        restored = spec.unflatten([-999])
        assert restored.value == -999

    def test_struct_map(self):
        class Answer(af.Struct):
            reasoning: str
            answer: int

        a = Answer(reasoning="abc", answer=42)
        mapped = af.utils.treelib.map(lambda x: f"[{x}]", a)
        assert mapped.reasoning == "[abc]"
        assert mapped.answer == "[42]"

    def test_nested_struct(self):
        class Inner(af.Struct):
            value: str

        class Outer(af.Struct):
            inner: Inner
            name: str

        o = Outer(inner=Inner(value="hello"), name="test")
        leaves = af.utils.treelib.leaves(o)
        assert leaves == ["hello", "test"]


class TestStructLmCall:
    def test_struct_lm_call_build(self):
        class Answer(af.Struct):
            reasoning: str
            answer: int

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Answer,
            )

        built_ir = af.build_ir(ir)("test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_params(self):
        class Answer(af.Struct):
            text: str

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o-mini",
                struct=Answer,
            )

        built_ir = af.build_ir(ir)("test")
        params = built_ir.ireqns[0].params
        assert params["model"] == "gpt-4o-mini"
        assert params["struct"] is Answer
        assert params["roles"] == ["user"]

    def test_struct_lm_call_eval_returns_var_tree(self):
        class Answer(af.Struct):
            field1: str
            field2: str

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Answer,
            )

        built_ir = af.build_ir(ir)("test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_pullback(self):
        class Answer(af.Struct):
            text: str

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Answer,
            )

        built_ir = af.build_ir(ir)("test")
        pb_ir = af.pullback(built_ir)
        assert pb_ir is not None
        assert len(pb_ir.ireqns) > 0

    def test_struct_lm_call_assertion_on_non_struct(self):
        class NotAStruct:
            pass

        try:
            af.struct_lm_call(
                [dict(role="user", content="test")],
                model="gpt-4o",
                struct=NotAStruct,
            )
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Struct" in str(e)

    def test_struct_lm_call_with_map_chain(self):
        class Step1(af.Struct):
            draft: str

        class Step2(af.Struct):
            final: str

        def ir(prompt: str):
            step1 = af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Step1,
            )
            refined = af.utils.treelib.map(lambda x: af.format("[refined] {}", x), step1)
            step2 = af.struct_lm_call(
                [dict(role="user", content=refined.draft)],
                model="gpt-4o",
                struct=Step2,
            )
            return step2

        built_ir = af.build_ir(ir)("test")
        prim_names = [eqn.prim.name for eqn in built_ir.ireqns]
        assert "struct_lm_call" in prim_names
        assert prim_names.count("struct_lm_call") == 2


class TestStructInAxes:
    def test_struct_as_in_axes(self):
        class Person(af.Struct):
            name: str
            sur: str

        def greet(p: Person) -> str:
            return af.format("Hello {}, {}", p.name, p.sur)

        ir = af.build_ir(greet)(Person(name="x", sur="y"))

        batch = af.batch(ir, in_axes=Person.model_construct(name=list, sur=None))

        result = af.call(batch)(
            # NOTE(asem): model_construct is used to bypass validation for axis spec
            Person.model_construct(name=["Alice", "Bob"], sur="Smith"),
        )
        assert result == ["Hello Alice, Smith", "Hello Bob, Smith"]

    def test_nested_struct_as_in_axes(self):
        class Inner(af.Struct):
            value: str

        class Outer(af.Struct):
            inner: Inner
            tag: str

        def process(o: Outer) -> str:
            return af.format("[{}] {}", o.tag, o.inner.value)

        ir = af.build_ir(process)(Outer(inner=Inner(value="x"), tag="t"))

        batch = af.batch(
            ir,
            # NOTE(asem): basically list is the container to batch over
            # and broadcast tag (None)
            in_axes=Outer.model_construct(inner=Inner.model_construct(value=list), tag=None),
        )

        result = af.call(batch)(
            Outer.model_construct(
                inner=Inner.model_construct(value=["a", "b", "c"]),
                tag="PREFIX",
            ),
        )
        assert result == ["[PREFIX] a", "[PREFIX] b", "[PREFIX] c"]

    def test_struct_hash_for_lru_cache(self):
        class A(af.Struct):
            x: str
            y: int

        a1 = A.model_construct(x=list, y=None)
        a2 = A.model_construct(x=list, y=None)

        hash(a1)
        hash(a2)

        assert hash(a1) == hash(a2)

    def test_batch_preserves_struct_output(self):
        class Output(af.Struct):
            first: str
            second: str

        def process(x: str) -> Output:
            return Output.model_construct(
                first=af.format("A:{}", x),
                second=af.format("B:{}", x),
            )

        ir = af.build_ir(process)("x")
        batch = af.batch(ir, in_axes=list)
        result = af.call(batch)(["1", "2", "3"])
        assert isinstance(result, Output)
        assert result.first == ["A:1", "A:2", "A:3"]
        assert result.second == ["B:1", "B:2", "B:3"]

    def test_batch_preserves_nested_struct_output(self):
        class Inner(af.Struct):
            value: str

        class Outer(af.Struct):
            inner: Inner
            tag: str

        def create(x: str) -> Outer:
            return Outer.model_construct(
                inner=Inner.model_construct(value=af.format("V:{}", x)),
                tag=af.format("T:{}", x),
            )

        ir = af.build_ir(create)("x")
        batch = af.batch(ir, in_axes=list)
        result = af.call(batch)(["a", "b"])
        assert isinstance(result, Outer)
        assert isinstance(result.inner, Inner)
        assert result.inner.value == ["V:a", "V:b"]
        assert result.tag == ["T:a", "T:b"]

    def test_batch_preserves_tuple_output(self):
        def dual(x: str) -> tuple[str, str]:
            return af.format("L:{}", x), af.format("R:{}", x)

        ir = af.build_ir(dual)("x")
        batch = af.batch(ir, in_axes=list)
        result = af.call(batch)(["a", "b"])
        assert result == (["L:a", "L:b"], ["R:a", "R:b"])

    def test_batch_preserves_nested_tuple_output(self):
        def nested(x: str) -> tuple[tuple[str, str], str]:
            return (af.format("A:{}", x), af.format("B:{}", x)), af.format("C:{}", x)

        ir = af.build_ir(nested)("x")
        batch = af.batch(ir, in_axes=list)
        result = af.call(batch)(["1", "2"])
        assert result == ((["A:1", "A:2"], ["B:1", "B:2"]), ["C:1", "C:2"])
