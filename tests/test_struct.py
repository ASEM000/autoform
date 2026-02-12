from typing import Annotated, Literal

import pytest
from annotated_types import Len

import autoform as af


class TestStructFieldValidation:
    def test_leaf_types_accepted(self):
        class Valid(af.Struct):
            a: str
            b: int
            c: float
            d: bool

    def test_nested_struct_accepted(self):
        class Inner(af.Struct):
            x: str

        class Outer(af.Struct):
            inner: Inner

    def test_literal_str_accepted(self):
        class WithLiteral(af.Struct):
            status: Literal["active", "inactive"]

    def test_literal_int_accepted(self):
        class WithLiteral(af.Struct):
            priority: Literal[1, 2, 3]

    def test_literal_bool_accepted(self):
        class WithLiteral(af.Struct):
            flag: Literal[True, False]

    def test_struct_field_accepted(self):
        class Inner(af.Struct):
            x: str

        class Outer(af.Struct):
            inner: Inner
            name: str

    def test_array_str_accepted(self):
        class WithArray(af.Struct):
            items: Annotated[list[str], Len(3, 3)]

    def test_array_int_accepted(self):
        class WithArray(af.Struct):
            scores: Annotated[list[int], Len(2, 2)]

    def test_array_struct_accepted(self):
        class Inner(af.Struct):
            x: str

        class WithArray(af.Struct):
            items: Annotated[list[Inner], Len(2, 2)]

    def test_array_literal_accepted(self):
        class WithArray(af.Struct):
            tags: Annotated[list[Literal["a", "b"]], Len(3, 3)]

    def test_mixed_literal_rejected(self):
        with pytest.raises(TypeError, match="Literal values must share one type"):

            class Bad(af.Struct):
                value: Literal["a", 1]

    def test_list_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                items: list[str]

    def test_dict_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                data: dict[str, int]

    def test_tuple_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                items: tuple[str, str]

    def test_optional_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                value: str | None


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

    def test_struct_with_array_leaves(self):
        class WithArray(af.Struct):
            name: str
            scores: Annotated[list[int], Len(3, 3)]

        a = WithArray(name="alice", scores=[10, 20, 30])
        leaves = af.utils.treelib.leaves(a)
        assert leaves == ["alice", 10, 20, 30]

    def test_struct_with_array_unflatten(self):
        class WithArray(af.Struct):
            tag: str
            items: Annotated[list[str], Len(2, 2)]

        a = WithArray(tag="x", items=["a", "b"])
        spec = af.utils.treelib.structure(a)
        restored = spec.unflatten(["y", "c", "d"])
        assert restored.tag == "y"
        assert restored.items == ["c", "d"]


class TestStructLmCall:
    def test_struct_lm_call_build(self):
        class Answer(af.Struct):
            reasoning: str
            answer: int

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=Answer,
            )

        built_ir = af.trace(ir)("test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_params(self):
        class Answer(af.Struct):
            text: str

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=Answer,
            )

        built_ir = af.trace(ir)("test")
        params = built_ir.ireqns[0].params
        assert params["model"] == "gpt-5.2"
        assert params["struct"] is Answer
        assert params["roles"] == ["user"]

    def test_struct_lm_call_eval_returns_var_tree(self):
        class Answer(af.Struct):
            field1: str
            field2: str

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=Answer,
            )

        built_ir = af.trace(ir)("test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_pullback(self):
        class Answer(af.Struct):
            text: str

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=Answer,
            )

        built_ir = af.trace(ir)("test")
        pb_ir = af.pullback(built_ir)
        assert pb_ir is not None
        assert len(pb_ir.ireqns) > 0

    def test_struct_lm_call_assertion_on_non_struct(self):
        class NotAStruct:
            pass

        try:
            af.struct_lm_call(
                [dict(role="user", content="test")],
                model="gpt-5.2",
                struct=NotAStruct,
            )
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Struct" in str(e)

    def test_struct_lm_call_with_array_field(self):
        class WithArray(af.Struct):
            items: Annotated[list[str], Len(3, 3)]

        def ir(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=WithArray,
            )

        built_ir = af.trace(ir)("test")
        assert len(built_ir.ireqns) == 1
        assert built_ir.ireqns[0].prim.name == "struct_lm_call"

    def test_struct_lm_call_with_map_chain(self):
        class Step1(af.Struct):
            draft: str

        class Step2(af.Struct):
            final: str

        def ir(prompt: str):
            step1 = af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=Step1,
            )
            refined = af.utils.treelib.map(lambda x: af.format("[refined] {}", x), step1)
            step2 = af.struct_lm_call(
                [dict(role="user", content=refined.draft)],
                model="gpt-5.2",
                struct=Step2,
            )
            return step2

        built_ir = af.trace(ir)("test")
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

        ir = af.trace(greet)(Person(name="x", sur="y"))

        batch = af.batch(ir, in_axes=Person.model_construct(name=True, sur=False))

        result = af.call(batch)(
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

        ir = af.trace(process)(Outer(inner=Inner(value="x"), tag="t"))

        batch = af.batch(
            ir,
            in_axes=Outer.model_construct(inner=Inner.model_construct(value=True), tag=False),
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

        a1 = A.model_construct(x=True, y=False)
        a2 = A.model_construct(x=True, y=False)

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

        ir = af.trace(process)("x")
        batch = af.batch(ir, in_axes=True)
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

        ir = af.trace(create)("x")
        batch = af.batch(ir, in_axes=True)
        result = af.call(batch)(["a", "b"])
        assert isinstance(result, Outer)
        assert isinstance(result.inner, Inner)
        assert result.inner.value == ["V:a", "V:b"]
        assert result.tag == ["T:a", "T:b"]

    def test_batch_preserves_tuple_output(self):
        def dual(x: str) -> tuple[str, str]:
            return af.format("L:{}", x), af.format("R:{}", x)

        ir = af.trace(dual)("x")
        batch = af.batch(ir, in_axes=True)
        result = af.call(batch)(["a", "b"])
        assert result == (["L:a", "L:b"], ["R:a", "R:b"])

    def test_batch_preserves_nested_tuple_output(self):
        def nested(x: str) -> tuple[tuple[str, str], str]:
            return (af.format("A:{}", x), af.format("B:{}", x)), af.format("C:{}", x)

        ir = af.trace(nested)("x")
        batch = af.batch(ir, in_axes=True)
        result = af.call(batch)(["1", "2"])
        assert result == ((["A:1", "A:2"], ["B:1", "B:2"]), ["C:1", "C:2"])


class TestStructFieldValidationErrors:
    def test_literal_non_leaf_base_rejected(self):
        with pytest.raises(TypeError, match="Literal base type must be str/int/float/bool"):

            class Bad(af.Struct):
                value: Literal[b"bytes"]

    def test_annotated_list_no_len_rejected(self):
        with pytest.raises(TypeError, match="No ``Len`` constraint"):

            class Bad(af.Struct):
                items: Annotated[list[str], "some marker"]

    def test_annotated_list_variable_len_rejected(self):
        with pytest.raises(TypeError, match="must be fixed size"):

            class Bad(af.Struct):
                items: Annotated[list[str], Len(1, 5)]

    def test_annotated_bare_list_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                items: Annotated[list, Len(3, 3)]

    def test_tuple_non_ellipsis_rejected(self):
        with pytest.raises(TypeError, match="tuple must be tuple\\[T, \\.\\.\\.\\] form"):

            class Bad(af.Struct):
                items: Annotated[tuple[str, int], Len(2, 2)]

    def test_tuple_ellipsis_accepted(self):
        class Valid(af.Struct):
            items: Annotated[tuple[str, ...], Len(3, 3)]

    def test_annotated_non_container_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                value: Annotated[str, "description"]

    def test_nested_container_bad_element_rejected(self):
        with pytest.raises(TypeError, match="invalid type"):

            class Bad(af.Struct):
                items: Annotated[list[dict], Len(2, 2)]


class TestTypeTree:
    def test_scalar_fields(self):
        class Simple(af.Struct):
            a: str
            b: int
            c: float
            d: bool

        tt = af.utils.struct_type_tree(Simple)
        assert isinstance(tt, Simple)
        assert tt.a is str
        assert tt.b is int
        assert tt.c is float
        assert tt.d is bool

    def test_nested_struct(self):
        class Inner(af.Struct):
            x: str

        class Outer(af.Struct):
            inner: Inner
            tag: int

        tt = af.utils.struct_type_tree(Outer)
        assert isinstance(tt, Outer)
        assert isinstance(tt.inner, Inner)
        assert tt.inner.x is str
        assert tt.tag is int

    def test_literal_resolves_to_base_type(self):
        class WithLiteral(af.Struct):
            status: Literal["a", "b"]
            priority: Literal[1, 2, 3]

        tt = af.utils.struct_type_tree(WithLiteral)
        assert tt.status is str
        assert tt.priority is int

    def test_list_container(self):
        class WithList(af.Struct):
            scores: Annotated[list[int], Len(3, 3)]

        tt = af.utils.struct_type_tree(WithList)
        assert isinstance(tt.scores, list)
        assert len(tt.scores) == 3
        assert all(t is int for t in tt.scores)

    def test_tuple_container(self):
        class WithTuple(af.Struct):
            tags: Annotated[tuple[str, ...], Len(2, 2)]

        tt = af.utils.struct_type_tree(WithTuple)
        assert isinstance(tt.tags, tuple)
        assert len(tt.tags) == 2
        assert all(t is str for t in tt.tags)

    def test_nested_struct_in_list(self):
        class Inner(af.Struct):
            v: str

        class Outer(af.Struct):
            items: Annotated[list[Inner], Len(2, 2)]

        tt = af.utils.struct_type_tree(Outer)
        assert isinstance(tt.items, list)
        assert len(tt.items) == 2
        assert all(isinstance(item, Inner) for item in tt.items)
        assert all(item.v is str for item in tt.items)

    def test_type_tree_map_produces_vars(self):
        class Answer(af.Struct):
            text: str
            score: int

        tt = af.utils.struct_type_tree(Answer)
        mapped = af.utils.treelib.map(lambda tp: f"Var({tp.__name__})", tt)
        assert isinstance(mapped, Answer)
        assert mapped.text == "Var(str)"
        assert mapped.score == "Var(int)"

    def test_type_tree_leaves_match_pytree_leaves(self):
        class Inner(af.Struct):
            x: str

        class Outer(af.Struct):
            inner: Inner
            scores: Annotated[list[int], Len(2, 2)]
            tag: str

        tt_leaves = af.utils.treelib.leaves(af.utils.struct_type_tree(Outer))
        assert tt_leaves == [str, int, int, str]

        # same count as real instance leaves
        real = Outer(inner=Inner(x="hi"), scores=[1, 2], tag="t")
        real_leaves = af.utils.treelib.leaves(real)
        assert len(tt_leaves) == len(real_leaves)


class TestStructEquality:
    def test_equal_structs(self):
        class Point(af.Struct):
            x: int
            y: int

        p1 = Point(x=1, y=2)
        p2 = Point(x=1, y=2)
        assert p1 == p2

    def test_unequal_values(self):
        class Point(af.Struct):
            x: int
            y: int

        p1 = Point(x=1, y=2)
        p2 = Point(x=1, y=3)
        assert p1 != p2

    def test_different_struct_types(self):
        class A(af.Struct):
            value: int

        class B(af.Struct):
            value: int

        a = A(value=1)
        b = B(value=1)
        assert a != b

    def test_struct_vs_non_struct(self):
        class Point(af.Struct):
            x: int
            y: int

        p = Point(x=1, y=2)
        assert p != (1, 2)
        assert p != {"x": 1, "y": 2}
        assert p != "not a struct"
        assert p != 42

    def test_nested_struct_equality(self):
        class Inner(af.Struct):
            v: str

        class Outer(af.Struct):
            inner: Inner
            name: str

        o1 = Outer(inner=Inner(v="hello"), name="test")
        o2 = Outer(inner=Inner(v="hello"), name="test")
        o3 = Outer(inner=Inner(v="world"), name="test")
        assert o1 == o2
        assert o1 != o3


class TestStructFrozen:
    def test_frozen_blocks_setattr(self):
        class Point(af.Struct):
            x: int
            y: int

        p = Point(x=1, y=2)
        with pytest.raises(Exception):
            p.x = 99

    def test_unflatten_bypasses_frozen(self):
        class Point(af.Struct):
            x: int
            y: int

        p = Point(x=1, y=2)
        spec = af.utils.treelib.structure(p)
        restored = spec.unflatten([10, 20])
        assert restored.x == 10
        assert restored.y == 20

    def test_model_construct_bypasses_frozen(self):
        class Point(af.Struct):
            x: int
            y: int

        p = Point.model_construct(x=True, y=False)
        assert p.x is True
        assert p.y is False

    def test_map_on_frozen_struct(self):
        class Pair(af.Struct):
            a: str
            b: str

        p = Pair(a="hello", b="world")
        mapped = af.utils.treelib.map(str.upper, p)
        assert mapped.a == "HELLO"
        assert mapped.b == "WORLD"


class TestStructComplex:
    def test_deeply_nested_3_levels(self):
        class L3(af.Struct):
            val: str

        class L2(af.Struct):
            child: L3
            tag: int

        class L1(af.Struct):
            nested: L2
            name: str

        obj = L1(nested=L2(child=L3(val="deep"), tag=42), name="root")
        leaves = af.utils.treelib.leaves(obj)
        assert leaves == ["deep", 42, "root"]

        spec = af.utils.treelib.structure(obj)
        restored = spec.unflatten(["new", 99, "top"])
        assert restored.nested.child.val == "new"
        assert restored.nested.tag == 99
        assert restored.name == "top"

    def test_struct_with_all_field_types(self):
        class Inner(af.Struct):
            v: str

        class Kitchen(af.Struct):
            scalar_str: str
            scalar_int: int
            scalar_float: float
            scalar_bool: bool
            literal_field: Literal["a", "b", "c"]
            list_field: Annotated[list[int], Len(2, 2)]
            nested: Inner

        obj = Kitchen(
            scalar_str="hello",
            scalar_int=1,
            scalar_float=3.14,
            scalar_bool=True,
            literal_field="a",
            list_field=[10, 20],
            nested=Inner(v="inner"),
        )
        leaves = af.utils.treelib.leaves(obj)
        assert leaves == ["hello", 1, 3.14, True, "a", 10, 20, "inner"]

    def test_type_tree_deeply_nested(self):
        class L3(af.Struct):
            x: str
            y: Annotated[list[int], Len(2, 2)]

        class L2(af.Struct):
            child: L3
            flag: Literal[True, False]

        class L1(af.Struct):
            nested: L2

        tt = af.utils.struct_type_tree(L1)
        tt_leaves = af.utils.treelib.leaves(tt)
        assert tt_leaves == [str, int, int, bool]

    def test_list_of_structs_type_tree(self):
        class Item(af.Struct):
            name: str
            score: int

        class Batch(af.Struct):
            items: Annotated[list[Item], Len(3, 3)]

        tt = af.utils.struct_type_tree(Batch)
        tt_leaves = af.utils.treelib.leaves(tt)
        assert tt_leaves == [str, int, str, int, str, int]

    def test_struct_roundtrip_flatten_unflatten(self):
        class Inner(af.Struct):
            x: str

        class Outer(af.Struct):
            a: int
            inner: Inner
            items: Annotated[list[str], Len(2, 2)]

        original = Outer(a=42, inner=Inner(x="hello"), items=["p", "q"])
        leaves, spec = af.utils.treelib.flatten(original)
        restored = spec.unflatten(leaves)
        assert original == restored

    def test_struct_trace_with_nested_access(self):
        class Inner(af.Struct):
            value: str

        class Outer(af.Struct):
            inner: Inner
            label: str

        def program(o: Outer) -> str:
            return af.format("{}: {}", o.label, o.inner.value)

        ir = af.trace(program)(Outer(inner=Inner(value="x"), label="y"))
        result = af.call(ir)(Outer(inner=Inner(value="world"), label="hello"))
        assert result == "hello: world"

    def test_struct_trace_with_list_field_access(self):
        class WithList(af.Struct):
            tag: str
            items: Annotated[list[str], Len(3, 3)]

        def program(w: WithList) -> str:
            return af.format("{}: {}, {}, {}", w.tag, w.items[0], w.items[1], w.items[2])

        ir = af.trace(program)(WithList(tag="t", items=["a", "b", "c"]))
        result = af.call(ir)(WithList(tag="label", items=["x", "y", "z"]))
        assert result == "label: x, y, z"

    def test_batch_struct_input_with_list_field(self):
        class Record(af.Struct):
            name: str
            tags: Annotated[list[str], Len(2, 2)]

        def summarize(r: Record) -> str:
            return af.format("{}={},{}", r.name, r.tags[0], r.tags[1])

        ir = af.trace(summarize)(Record(name="x", tags=["a", "b"]))
        batch = af.batch(
            ir,
            in_axes=Record.model_construct(name=True, tags=[False, False]),
        )
        result = af.call(batch)(
            Record.model_construct(name=["Alice", "Bob"], tags=["p", "q"]),
        )
        assert result == ["Alice=p,q", "Bob=p,q"]

    def test_pushforward_through_nested_struct(self):
        class Pair(af.Struct):
            a: str
            b: str

        def program(p: Pair) -> str:
            return af.format("[{},{}]", p.a, p.b)

        ir = af.trace(program)(Pair(a="x", b="y"))
        pf_ir = af.pushforward(ir)
        primal, tangent = af.call(pf_ir)((
            Pair(a="hello", b="world"),
            Pair(a="delta_a", b="delta_b"),
        ))
        assert primal == "[hello,world]"
        assert tangent == "[delta_a,delta_b]"

    def test_pullback_through_nested_struct(self):
        class Pair(af.Struct):
            a: str
            b: str

        def program(p: Pair) -> str:
            return af.format("[{},{}]", p.a, p.b)

        ir = af.trace(program)(Pair(a="x", b="y"))
        pb_ir = af.pullback(ir)
        output, grad = af.call(pb_ir)((Pair(a="hello", b="world"), "feedback"))
        assert output == "[hello,world]"
        assert isinstance(grad, Pair)

    def test_struct_lm_call_nested_struct_traces(self):
        class Inner(af.Struct):
            detail: str

        class Outer(af.Struct):
            summary: str
            inner: Inner
            tags: Annotated[list[str], Len(2, 2)]

        def program(prompt: str):
            return af.struct_lm_call(
                [dict(role="user", content=prompt)],
                model="gpt-5.2",
                struct=Outer,
            )

        ir = af.trace(program)("test")
        assert len(ir.ireqns) == 1
        tt_leaves = af.utils.treelib.leaves(af.utils.struct_type_tree(Outer))
        assert tt_leaves == [str, str, str, str]
