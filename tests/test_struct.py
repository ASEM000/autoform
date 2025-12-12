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
        assert params["roles"] == ("user",)

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
