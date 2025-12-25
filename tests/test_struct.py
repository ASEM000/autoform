import autoform.core as core


class TestStruct:
    def test_struct_is_pytree(self):
        class Answer(core.Struct):
            reasoning: str
            answer: int

        a = Answer(reasoning="think step by step", answer=42)
        leaves = core.treelib.leaves(a)
        assert leaves == ["think step by step", 42]

    def test_struct_unflatten(self):
        class Answer(core.Struct):
            reasoning: str
            answer: int

        a = Answer(reasoning="original", answer=1)
        spec = core.treelib.structure(a)
        restored = spec.unflatten(["new reasoning", 100])
        assert restored.reasoning == "new reasoning"
        assert restored.answer == 100
        assert isinstance(restored, Answer)

    def test_struct_unflatten_skips_validation(self):
        class Positive(core.Struct):
            value: int

        spec = core.treelib.structure(Positive(value=1))
        restored = spec.unflatten([-999])
        assert restored.value == -999

    def test_struct_map(self):
        class Answer(core.Struct):
            reasoning: str
            answer: int

        a = Answer(reasoning="abc", answer=42)
        mapped = core.treelib.map(lambda x: f"[{x}]", a)
        assert mapped.reasoning == "[abc]"
        assert mapped.answer == "[42]"

    def test_nested_struct(self):
        class Inner(core.Struct):
            value: str

        class Outer(core.Struct):
            inner: Inner
            name: str

        o = Outer(inner=Inner(value="hello"), name="test")
        leaves = core.treelib.leaves(o)
        assert leaves == ["hello", "test"]


class TestStructLmCall:
    def test_struct_lm_call_ir_build(self):
        class Answer(core.Struct):
            reasoning: str
            answer: int

        def ir(prompt: str):
            return core.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Answer,
            )

        built_ir = core.build_ir(ir, "test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_params(self):
        class Answer(core.Struct):
            text: str

        def ir(prompt: str):
            return core.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o-mini",
                struct=Answer,
            )

        built_ir = core.build_ir(ir, "test")
        params = built_ir.ireqns[0].params
        assert params["model"] == "gpt-4o-mini"
        assert params["struct"] is Answer
        assert params["roles"] == ["user"]

    def test_struct_lm_call_eval_returns_var_tree(self):
        class Answer(core.Struct):
            field1: str
            field2: str

        def ir(prompt: str):
            return core.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Answer,
            )

        built_ir = core.build_ir(ir, "test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_pullback_ir(self):
        class Answer(core.Struct):
            text: str

        def ir(prompt: str):
            return core.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Answer,
            )

        built_ir = core.build_ir(ir, "test")
        pb_ir = core.pullback_ir(built_ir)
        assert pb_ir is not None
        assert len(pb_ir.ireqns) > 0

    def test_struct_lm_call_assertion_on_non_struct(self):
        class NotAStruct:
            pass

        try:
            core.struct_lm_call(
                [dict(role="user", content="test")],
                model="gpt-4o",
                struct=NotAStruct,
            )
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Struct" in str(e)

    def test_struct_lm_call_with_map_chain(self):
        class Step1(core.Struct):
            draft: str

        class Step2(core.Struct):
            final: str

        def ir(prompt: str):
            step1 = core.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-4o",
                struct=Step1,
            )
            refined = core.treelib.map(lambda x: core.format("[refined] {}", x), step1)
            step2 = core.struct_lm_call(
                [dict(role="user", content=refined.draft)],
                model="gpt-4o",
                struct=Step2,
            )
            return step2

        built_ir = core.build_ir(ir, "test")
        prim_names = [eqn.prim.name for eqn in built_ir.ireqns]
        assert "struct_lm_call" in prim_names
        assert prim_names.count("struct_lm_call") == 2


class TestStructInAxes:
    def test_struct_as_in_axes(self):
        class Person(core.Struct):
            name: str
            sur: str

        def greet(p: Person) -> str:
            return core.format("Hello {}, {}", p.name, p.sur)

        ir = core.build_ir(greet, Person(name="x", sur="y"))

        batch_ir = core.batch_ir(ir, in_axes=Person.model_construct(name=list, sur=None))

        result = core.run_ir(
            batch_ir,
            # NOTE(asem): model_construct is used to bypass validation for axis spec
            Person.model_construct(name=["Alice", "Bob"], sur="Smith"),
        )
        assert result == ["Hello Alice, Smith", "Hello Bob, Smith"]

    def test_nested_struct_as_in_axes(self):
        class Inner(core.Struct):
            value: str

        class Outer(core.Struct):
            inner: Inner
            tag: str

        def process(o: Outer) -> str:
            return core.format("[{}] {}", o.tag, o.inner.value)

        ir = core.build_ir(process, Outer(inner=Inner(value="x"), tag="t"))

        batch_ir = core.batch_ir(
            ir,
            # NOTE(asem): basically list is the container to batch over
            # and broadcast tag (None)
            in_axes=Outer.model_construct(inner=Inner.model_construct(value=list), tag=None),
        )

        result = core.run_ir(
            batch_ir,
            Outer.model_construct(
                inner=Inner.model_construct(value=["a", "b", "c"]),
                tag="PREFIX",
            ),
        )
        assert result == ["[PREFIX] a", "[PREFIX] b", "[PREFIX] c"]

    def test_struct_hash_for_lru_cache(self):
        class A(core.Struct):
            x: str
            y: int

        a1 = A.model_construct(x=list, y=None)
        a2 = A.model_construct(x=list, y=None)

        hash(a1)
        hash(a2)

        assert hash(a1) == hash(a2)

    def test_batch_preserves_struct_output(self):
        class Output(core.Struct):
            first: str
            second: str

        def process(x: str) -> Output:
            return Output.model_construct(
                first=core.format("A:{}", x),
                second=core.format("B:{}", x),
            )

        ir = core.build_ir(process, "x")
        batch_ir = core.batch_ir(ir, in_axes=list)
        result = core.run_ir(batch_ir, ["1", "2", "3"])
        assert isinstance(result, Output)
        assert result.first == ["A:1", "A:2", "A:3"]
        assert result.second == ["B:1", "B:2", "B:3"]

    def test_batch_preserves_nested_struct_output(self):
        class Inner(core.Struct):
            value: str

        class Outer(core.Struct):
            inner: Inner
            tag: str

        def create(x: str) -> Outer:
            return Outer.model_construct(
                inner=Inner.model_construct(value=core.format("V:{}", x)),
                tag=core.format("T:{}", x),
            )

        ir = core.build_ir(create, "x")
        batch_ir = core.batch_ir(ir, in_axes=list)
        result = core.run_ir(batch_ir, ["a", "b"])
        assert isinstance(result, Outer)
        assert isinstance(result.inner, Inner)
        assert result.inner.value == ["V:a", "V:b"]
        assert result.tag == ["T:a", "T:b"]

    def test_batch_preserves_tuple_output(self):
        def dual(x: str) -> tuple[str, str]:
            return core.format("L:{}", x), core.format("R:{}", x)

        ir = core.build_ir(dual, "x")
        batch_ir = core.batch_ir(ir, in_axes=list)
        result = core.run_ir(batch_ir, ["a", "b"])
        assert result == (["L:a", "L:b"], ["R:a", "R:b"])

    def test_batch_preserves_nested_tuple_output(self):
        def nested(x: str) -> tuple[tuple[str, str], str]:
            return (core.format("A:{}", x), core.format("B:{}", x)), core.format("C:{}", x)

        ir = core.build_ir(nested, "x")
        batch_ir = core.batch_ir(ir, in_axes=list)
        result = core.run_ir(batch_ir, ["1", "2"])
        assert result == ((["A:1", "A:2"], ["B:1", "B:2"]), ["C:1", "C:2"])
