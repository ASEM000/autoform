import autoform.core as core


class TestFoldIR:
    def test_folds_constant_format(self):
        def program(x):
            constant = core.format("{}, {}", "hello", "world")
            return core.concat(x, constant)

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(folded.ireqns) == 1
        assert folded.ireqns[0].prim.name == "concat"

    def test_folds_constant_concat(self):
        def program(x):
            constant = core.concat("hello", " world")
            return core.concat(x, constant)

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(folded.ireqns) == 1
        assert folded.ireqns[0].prim.name == "concat"

    def test_keeps_non_constant_equation(self):
        def program(x):
            return core.concat(x, "!")

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 1
        assert len(folded.ireqns) == 1

    def test_propagates_folded_values(self):
        def program(x):
            a = core.format("{}", "constant")
            b = core.concat(a, " suffix")
            return core.concat(x, b)

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 3
        assert len(folded.ireqns) == 1
        assert folded.ireqns[0].prim.name == "concat"

    def test_partial_fold_chain(self):
        def program(x):
            a = core.concat("hello", " ")
            b = core.concat(a, x)
            c = core.concat(b, "!")
            return c

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 3
        assert len(folded.ireqns) == 2

    def test_output_substitution(self):
        def program(x):
            return core.concat("hello", " world")

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 1
        assert len(folded.ireqns) == 0

        out_leaves = core.treelib.leaves(folded.out_irtree)
        assert len(out_leaves) == 1
        assert core.is_irlit(out_leaves[0])
        assert out_leaves[0].value == "hello world"

    def test_identity_when_no_constants(self):
        def program(x, y):
            return core.concat(x, y)

        ir = core.build_ir(program, "a", "b")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == len(folded.ireqns)

    def test_empty_ir(self):
        def program(x):
            return x

        ir = core.build_ir(program, "input")
        folded = core.fold_ir(ir)

        assert len(ir.ireqns) == 0
        assert len(folded.ireqns) == 0


class TestFoldIRExecution:
    def test_folded_ir_executes_correctly(self):
        def program(x):
            prefix = core.concat("Hello", ", ")
            return core.concat(prefix, x)

        ir = core.build_ir(program, "World")
        folded = core.fold_ir(ir)

        original_result = core.run_ir(ir, "World")
        folded_result = core.run_ir(folded, "World")

        assert original_result == folded_result == "Hello, World"

    def test_fully_constant_ir_executes(self):
        def program(x):
            return core.concat("hello", " world")

        ir = core.build_ir(program, "ignored")
        folded = core.fold_ir(ir)

        result = core.run_ir(folded, "anything")
        assert result == "hello world"
