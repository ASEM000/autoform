import autoform as af


class TestFoldIR:
    def test_folds_constant_format(self):
        def program(x):
            constant = af.format("{}, {}", "hello", "world")
            return af.concat(x, constant)

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 2
        assert len(folded.ireqns) == 1
        assert folded.ireqns[0].prim.name == "concat"

    def test_folds_constant_concat(self):
        def program(x):
            constant = af.concat("hello", " world")
            return af.concat(x, constant)

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 2
        assert len(folded.ireqns) == 1
        assert folded.ireqns[0].prim.name == "concat"

    def test_keeps_non_constant_equation(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 1
        assert len(folded.ireqns) == 1

    def test_propagates_folded_values(self):
        def program(x):
            a = af.format("{}", "constant")
            b = af.concat(a, " suffix")
            return af.concat(x, b)

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 3
        assert len(folded.ireqns) == 1
        assert folded.ireqns[0].prim.name == "concat"

    def test_partial_fold_chain(self):
        def program(x):
            a = af.concat("hello", " ")
            b = af.concat(a, x)
            c = af.concat(b, "!")
            return c

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 3
        assert len(folded.ireqns) == 2

    def test_output_substitution(self):
        def program(x):
            return af.concat("hello", " world")

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 1
        assert len(folded.ireqns) == 0

        out_leaves = af.utils.treelib.leaves(folded.out_irtree)
        assert len(out_leaves) == 1
        assert isinstance(out_leaves[0], af.core.IRLit)
        assert out_leaves[0].value == "hello world"

    def test_identity_when_no_constants(self):
        def program(x, y):
            return af.concat(x, y)

        ir = af.build_ir(program)("a", "b")
        folded = af.fold(ir)

        assert len(ir.ireqns) == len(folded.ireqns)

    def test_empty_ir(self):
        def program(x):
            return x

        ir = af.build_ir(program)("input")
        folded = af.fold(ir)

        assert len(ir.ireqns) == 0
        assert len(folded.ireqns) == 0


class TestFoldIRExecution:
    def test_folded_ir_executes_correctly(self):
        def program(x):
            prefix = af.concat("Hello", ", ")
            return af.concat(prefix, x)

        ir = af.build_ir(program)("World")
        folded = af.fold(ir)

        original_result = af.call(ir)("World")
        folded_result = af.call(folded)("World")

        assert original_result == folded_result == "Hello, World"

    def test_fully_constant_ir_executes(self):
        def program(x):
            return af.concat("hello", " world")

        ir = af.build_ir(program)("ignored")
        folded = af.fold(ir)

        result = af.call(folded)("anything")
        assert result == "hello world"
