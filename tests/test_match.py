"""Tests for pattern matching and replace on IR structures."""

import autoform.core as core


class TestIREqnMatchArgs:
    def test_match_by_primitive(self):
        def func(x):
            return core.concat("Hello, ", x)

        ir = core.build_ir(func, "world")
        eqn = ir.ireqns[0]

        match eqn:
            case core.IREqn(prim=p) if p == core.concat_p:
                matched = True
            case _:
                matched = False

        assert matched

    def test_match_by_params(self):
        def func(x):
            return core.mark(x, tag="step1")

        ir = core.build_ir(func, "test")
        eqn = ir.ireqns[0]

        match eqn:
            case core.IREqn(params={"tag": tag}):
                matched_tag = tag
            case _:
                matched_tag = None

        assert matched_tag == "step1"

    def test_match_positional_destructuring(self):
        def func(x):
            return core.format("Value: {}", x)

        ir = core.build_ir(func, "test")
        eqn = ir.ireqns[0]

        match eqn:
            case core.IREqn(prim, in_tree, out_tree, params):
                assert prim == core.format_p
                assert params["template"] == "Value: {}"

    def test_match_in_loop(self):
        def func(x):
            a = core.mark(x, tag="step1")
            b = core.concat(a, "!")
            c = core.mark(b, tag="step2")
            return c

        ir = core.build_ir(func, "test")

        tags_found = []
        for eqn in ir.ireqns:
            match eqn:
                case core.IREqn(prim=p, params={"tag": tag}) if p == core.mark_p:
                    tags_found.append(tag)

        assert tags_found == ["step1", "step2"]

    def test_match_and_transform(self):
        """Test matching equations and building a transformed IR."""

        def func(x):
            a = core.mark(x, tag="old_tag")
            return core.concat(a, "!")

        ir = core.build_ir(func, "test")

        # Transform: find mark with old_tag and change to new_tag
        new_eqns = []
        for eqn in ir.ireqns:
            match eqn:
                case core.IREqn(prim=p, params={"tag": "old_tag"}) if p == core.mark_p:
                    new_eqns.append(eqn.replace(params={"tag": "new_tag"}))
                case _:
                    new_eqns.append(eqn)

        new_ir = core.IR(
            ireqns=new_eqns,
            in_ir_tree=ir.in_ir_tree,
            out_ir_tree=ir.out_ir_tree,
        )

        # Verify the tag was changed
        assert new_ir.ireqns[0].params["tag"] == "new_tag"
        # Verify the IR still works
        result = core.run_ir(new_ir, "hello")
        assert result == "hello!"


class TestIREqnReplace:
    def test_replace_params(self):
        def func(x):
            return core.mark(x, tag="old")

        ir = core.build_ir(func, "test")
        eqn = ir.ireqns[0]

        new_eqn = eqn.replace(params={"tag": "new"})

        assert eqn.params["tag"] == "old"  # original unchanged
        assert new_eqn.params["tag"] == "new"
        assert new_eqn.prim == eqn.prim
        assert new_eqn.in_ir_tree == eqn.in_ir_tree
        assert new_eqn.out_ir_tree == eqn.out_ir_tree

    def test_replace_prim(self):
        def func(x):
            return core.concat(x, "!")

        ir = core.build_ir(func, "test")
        eqn = ir.ireqns[0]

        new_eqn = eqn.replace(prim=core.format_p)

        assert eqn.prim == core.concat_p  # original unchanged
        assert new_eqn.prim == core.format_p

    def test_replace_preserves_unspecified(self):
        def func(x):
            return core.mark(x, tag="test")

        ir = core.build_ir(func, "test")
        eqn = ir.ireqns[0]

        # Replace only params, everything else should be preserved
        new_eqn = eqn.replace(params={"tag": "changed"})

        assert new_eqn.prim is eqn.prim
        assert new_eqn.in_ir_tree is eqn.in_ir_tree
        assert new_eqn.out_ir_tree is eqn.out_ir_tree


class TestInsertAfterPattern:
    def test_insert_equation_after_match(self):
        """Test inserting a new equation after a matched one."""

        def func(x):
            a = core.mark(x, tag="insert_here")
            return core.concat(a, "!")

        ir = core.build_ir(func, "test")
        assert len(ir.ireqns) == 2

        # Insert a new mark after the first mark
        new_eqns = []
        for eqn in ir.ireqns:
            new_eqns.append(eqn)
            match eqn:
                case core.IREqn(prim=p, params={"tag": "insert_here"}) if p == core.mark_p:
                    # Create a new mark equation with the same IO but different tag
                    inserted = eqn.replace(params={"tag": "inserted"})
                    new_eqns.append(inserted)

        new_ir = core.IR(
            ireqns=new_eqns,
            in_ir_tree=ir.in_ir_tree,
            out_ir_tree=ir.out_ir_tree,
        )

        assert len(new_ir.ireqns) == 3
        assert new_ir.ireqns[0].params["tag"] == "insert_here"
        assert new_ir.ireqns[1].params["tag"] == "inserted"
        assert new_ir.ireqns[2].prim == core.concat_p
