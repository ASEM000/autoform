import autoform as af
from autoform.control import ControlTag
from autoform.harvest import CheckpointEffect
from autoform.string import StringTag


class TestIREqnMatchArgs:
    def test_match_by_primitive(self):
        def func(x):
            return af.concat("Hello, ", x)

        ir = af.trace(func)("world")
        eqn = ir.ireqns[0]

        match eqn:
            case af.core.IREqn(prim=p) if p == af.string.concat_p:
                matched = True
            case _:
                matched = False

        assert matched

    def test_match_by_params(self):
        def func(x):
            return af.checkpoint(x, key="x", collection="step1")

        ir = af.trace(func)("test")
        eqn = ir.ireqns[0]

        match eqn.effect:
            case CheckpointEffect(key=key, collection=collection):
                matched_tag = collection
            case _:
                matched_tag = None

        assert matched_tag == "step1"

    def test_match_positional_destructuring(self):
        def func(x):
            return af.format("Value: {}", x)

        ir = af.trace(func)("test")
        eqn = ir.ireqns[0]

        match eqn:
            case af.core.IREqn(prim, in_tree, out_tree, effect, params):
                assert prim == af.string.format_p
                assert params["template"] == "Value: {}"

    def test_match_in_loop(self):
        def func(x):
            a = af.checkpoint(x, key="a", collection="step1")
            b = af.concat(a, "!")
            c = af.checkpoint(b, key="c", collection="step2")
            return c

        ir = af.trace(func)("test")

        tags_found = []
        for eqn in ir.ireqns:
            match eqn.effect:
                case CheckpointEffect(key=_, collection=collection):
                    tags_found.append(collection)

        assert tags_found == ["step1", "step2"]

    def test_match_and_transform(self):
        def func(x):
            a = af.checkpoint(x, key="a", collection="old_tag")
            return af.concat(a, "!")

        ir = af.trace(func)("test")

        new_eqns = []
        for eqn in ir.ireqns:
            match eqn.effect:
                case CheckpointEffect(key=key, collection="old_tag"):
                    new_effect = CheckpointEffect(key=key, collection="new_tag")
                    new_eqns.append(
                        af.core.IREqn(
                            eqn.prim, new_effect, eqn.in_irtree, eqn.out_irtree, eqn.params
                        )
                    )
                case _:
                    new_eqns.append(eqn)

        new_ir = af.core.IR(
            ireqns=new_eqns,
            in_irtree=ir.in_irtree,
            out_irtree=ir.out_irtree,
        )

        assert new_ir.ireqns[0].effect.collection == "new_tag"

        result = af.call(new_ir)("hello")
        assert result == "hello!"


class TestIREqnWithParams:
    def test_using_merges(self):
        def func(x):
            return af.checkpoint(x, key="x", collection="old")

        ir = af.trace(func)("test")
        eqn = ir.ireqns[0]

        new_effect = CheckpointEffect(key="x", collection="new")
        new_eqn = af.core.IREqn(eqn.prim, new_effect, eqn.in_irtree, eqn.out_irtree, eqn.params)

        assert eqn.effect.collection == "old"
        assert new_eqn.effect.collection == "new"
        assert new_eqn.prim == eqn.prim
        assert new_eqn.in_irtree == eqn.in_irtree
        assert new_eqn.out_irtree == eqn.out_irtree

    def test_using_preserves_fields(self):
        def func(x):
            return af.checkpoint(x, key="x", collection="test")

        ir = af.trace(func)("test")
        eqn = ir.ireqns[0]

        new_eqn = eqn.using(collection="changed")

        assert new_eqn.prim is eqn.prim
        assert new_eqn.in_irtree is eqn.in_irtree
        assert new_eqn.out_irtree is eqn.out_irtree


class TestInsertAfterPattern:
    def test_insert_equation_after_match(self):
        def func(x):
            a = af.checkpoint(x, key="a", collection="insert_here")
            return af.concat(a, "!")

        ir = af.trace(func)("test")
        assert len(ir.ireqns) == 2

        new_eqns = []
        for eqn in ir.ireqns:
            new_eqns.append(eqn)
            match eqn.effect:
                case CheckpointEffect(key=key, collection="insert_here"):
                    new_effect = CheckpointEffect(key=key, collection="inserted")
                    inserted = af.core.IREqn(
                        eqn.prim, new_effect, eqn.in_irtree, eqn.out_irtree, eqn.params
                    )
                    new_eqns.append(inserted)

        new_ir = af.core.IR(
            ireqns=new_eqns,
            in_irtree=ir.in_irtree,
            out_irtree=ir.out_irtree,
        )

        assert len(new_ir.ireqns) == 3
        assert new_ir.ireqns[0].effect.collection == "insert_here"
        assert new_ir.ireqns[1].effect.collection == "inserted"
        assert new_ir.ireqns[2].prim == af.string.concat_p


class TestIRMatchArgs:
    def test_match_ir_positional(self):
        ir = af.trace(lambda x: af.concat("a", x))("b")

        match ir:
            case af.core.IR(eqns, in_tree, out_tree):
                assert len(eqns) == 1
                assert isinstance(in_tree, af.core.IRVar)
                assert isinstance(out_tree, af.core.IRVar)
            case _:
                assert False, "Pattern should match"

    def test_match_ir_with_nested_eqns(self):
        ir = af.trace(lambda x: af.concat("a", x))("b")

        match ir:
            case af.core.IR([af.core.IREqn(prim, effect, in_tree, out_tree, params)], _, _):
                assert prim == af.string.concat_p
                assert StringTag in prim.tag
                assert len(af.utils.treelib.leaves(in_tree)) == 2
            case _:
                assert False, "Pattern should match single equation"

    def test_match_multiple_equations(self):
        def program(x, y):
            formatted = af.format("Hello {}", x)
            return af.concat(formatted, y)

        ir = af.trace(program)("World", "!")

        match ir:
            case af.core.IR([af.core.IREqn(p1, _, _, _), af.core.IREqn(p2, _, _, _)], _, _):
                assert p1.name == "format"
                assert p2.name == "concat"
            case _:
                assert False, "Pattern should match two equations"

    def test_match_by_primitive_tag(self):
        ir = af.trace(lambda x: af.concat("a", x))("b")

        match ir:
            case af.core.IR([af.core.IREqn(af.core.Primitive(name, tag), _, _, _)], _, _) if (
                StringTag in tag
            ):
                assert name == "concat"
            case _:
                assert False, "Pattern should match string-tagged primitive"

    def test_match_higher_order_primitive_with_nested_ir(self):
        inner_ir = af.trace(lambda x: af.concat("a", x))("b")
        pf_ir = af.pushforward(inner_ir)

        match pf_ir:
            case af.core.IR(
                [af.core.IREqn(af.core.Primitive("pushforward_call", _), _, _, effect, params)],
                _,
                _,
            ):
                nested = params["ir"]
                assert isinstance(nested, af.core.IR)
                assert len(nested.ireqns) == 1
            case _:
                assert False, "Pattern should match pushforward_call"

    def test_match_switch_branches(self):
        branches = {
            "a": af.trace(lambda x: af.concat("A: ", x))("X"),
            "b": af.trace(lambda x: af.concat("B: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "test")

        match ir:
            case af.core.IR(
                [af.core.IREqn(af.core.Primitive("switch", tag), _, _, effect, params)], _, _
            ) if ControlTag in tag:
                branch_dict = params["branches"]
                assert "a" in branch_dict
                assert "b" in branch_dict
                assert isinstance(branch_dict["a"], af.core.IR)
            case _:
                assert False, "Pattern should match switch"

    def test_match_ir_guard(self):
        ir = af.trace(lambda x: af.concat("a", x))("b")

        match ir:
            case af.core.IR(eqns, _, _) if len(eqns) == 1:
                single_eqn = True
            case af.core.IR(eqns, _, _) if len(eqns) > 1:
                single_eqn = False
            case _:
                single_eqn = None

        assert single_eqn is True

    def test_match_all_string_primitives(self):
        def program(x, y):
            a = af.format("Hello {}", x)
            return af.concat(a, y)

        ir = af.trace(program)("World", "!")

        match ir:
            case af.core.IR(eqns, _, _) if all(StringTag in e.prim.tag for e in eqns):
                all_string = True
            case _:
                all_string = False

        assert all_string is True
