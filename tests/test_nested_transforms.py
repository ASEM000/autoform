import autoform as af


class TestBatchOfPushforward:
    def test_batch_of_pushforward_ir(self):
        def program(x):
            y = af.format("Value: {}", x)
            z = af.concat(y, "!")
            return z

        ir = af.build_ir(program, "x")
        pf_ir = af.pushforward_ir(ir)
        batch_pf_ir = af.batch_ir(pf_ir, in_axes=(list, list))
        primals = ["a", "b", "c"]
        tangents = ["da", "db", "dc"]
        result = af.run_ir(batch_pf_ir, primals, tangents)
        assert result == (
            ["Value: a!", "Value: b!", "Value: c!"],
            ["Value: da!", "Value: db!", "Value: dc!"],
        )

    def test_batch_of_pushforward_single_element(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program, "x")
        pf_ir = af.pushforward_ir(ir)
        batch_pf_ir = af.batch_ir(pf_ir, in_axes=(list, list))
        result = af.run_ir(batch_pf_ir, ["a"], ["da"])
        assert result == (["a!"], ["da!"])


class TestBatchOfPullback:
    def test_batch_of_pullback_ir(self):
        def program(x):
            y = af.format("Value: {}", x)
            z = af.concat(y, "!")
            return z

        ir = af.build_ir(program, "x")
        pb_ir = af.pullback_ir(ir)
        batch_pb_ir = af.batch_ir(pb_ir, in_axes=(list, list))
        primals = ["a", "b", "c"]
        cotangents = ["g1", "g2", "g3"]
        result = af.run_ir(batch_pb_ir, primals, cotangents)
        assert result == (
            ["Value: a!", "Value: b!", "Value: c!"],
            ["g1", "g2", "g3"],
        )

    def test_batch_of_pullback_single_element(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program, "x")
        pb_ir = af.pullback_ir(ir)
        batch_pb_ir = af.batch_ir(pb_ir, in_axes=(list, list))
        result = af.run_ir(batch_pb_ir, ["a"], ["g"])
        assert result == (["a!"], ["g"])


class TestPushforwardOfBatch:
    def test_pushforward_of_batch_ir(self):
        def program(x):
            y = af.format("Value: {}", x)
            z = af.concat(y, "!")
            return z

        ir = af.build_ir(program, "x")
        batch_ir_obj = af.batch_ir(ir)
        pf_batch_ir = af.pushforward_ir(batch_ir_obj)
        p_xs = ["a", "b"]
        t_xs = ["da", "db"]
        result = af.run_ir(pf_batch_ir, (p_xs, t_xs))
        assert result == (
            ["Value: a!", "Value: b!"],
            ["Value: da!", "Value: db!"],
        )

    def test_pushforward_of_batch_single_element(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program, "x")
        batch_ir_obj = af.batch_ir(ir)
        pf_batch_ir = af.pushforward_ir(batch_ir_obj)
        result = af.run_ir(pf_batch_ir, (["a"], ["da"]))
        assert result == (["a!"], ["da!"])


class TestPullbackOfBatch:
    def test_pullback_of_batch_ir(self):
        def program(x):
            y = af.format("Value: {}", x)
            z = af.concat(y, "!")
            return z

        ir = af.build_ir(program, "x")
        batch_ir_obj = af.batch_ir(ir)
        pb_batch_ir = af.pullback_ir(batch_ir_obj)
        p_xs = ["a", "b"]
        cotangent_out = ["g1", "g2"]
        result = af.run_ir(pb_batch_ir, (p_xs, cotangent_out))
        assert result == (
            ["Value: a!", "Value: b!"],
            ["g1", "g2"],
        )

    def test_pullback_of_batch_single_element(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program, "x")
        batch_ir_obj = af.batch_ir(ir)
        pb_batch_ir = af.pullback_ir(batch_ir_obj)
        result = af.run_ir(pb_batch_ir, (["a"], ["g"]))
        assert result == (["a!"], ["g"])


class TestTripleNesting:
    def test_batch_of_batch(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program, "x")
        batch_ir1 = af.batch_ir(ir)
        batch_ir2 = af.batch_ir(batch_ir1)
        inputs = [["a", "b"], ["c", "d"]]
        result = af.run_ir(batch_ir2, inputs)
        assert result == [["a!", "b!"], ["c!", "d!"]]

    def test_pushforward_of_pushforward_of_batch(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program, "x")
        batch_ir_obj = af.batch_ir(ir)
        pf1 = af.pushforward_ir(batch_ir_obj)
        pf2 = af.pushforward_ir(pf1)
        p_xs = ["a", "b"]
        t1_xs = ["t1a", "t1b"]
        t2_xs = (["t2a", "t2b"], ["t2t1a", "t2t1b"])
        result = af.run_ir(pf2, ((p_xs, t1_xs), t2_xs))
        assert result == (
            (["a!", "b!"], ["t1a!", "t1b!"]),
            (["t2a!", "t2b!"], ["t2t1a!", "t2t1b!"]),
        )


class TestTripleBatch:
    def test_triple_batch(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        b3 = af.batch_ir(b2)
        inputs = [[["a", "b"], ["c"]], [["d", "e", "f"]]]
        result = af.run_ir(b3, inputs)
        assert result == [[["a!", "b!"], ["c!"]], [["d!", "e!", "f!"]]]

    def test_quadruple_batch(self):
        def f(x):
            return af.format("[{}]", x)

        ir = af.build_ir(f, "x")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        b3 = af.batch_ir(b2)
        b4 = af.batch_ir(b3)
        inputs = [[[["a"]]]]
        result = af.run_ir(b4, inputs)
        assert result == [[[["[a]"]]]]


class TestTriplePushforward:
    def test_triple_pushforward(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        pf1 = af.pushforward_ir(ir)
        pf2 = af.pushforward_ir(pf1)
        pf3 = af.pushforward_ir(pf2)
        p = "a"
        t1 = "t1"
        t2 = ("t2p", "t2t")
        t3 = (("t3pp", "t3pt"), ("t3tp", "t3tt"))
        result = af.run_ir(pf3, (((p, t1), t2), t3))
        assert result == (
            (("a!", "t1!"), ("t2p!", "t2t!")),
            (("t3pp!", "t3pt!"), ("t3tp!", "t3tt!")),
        )

    def test_quadruple_pushforward(self):
        def f(x):
            return af.format("[{}]", x)

        ir = af.build_ir(f, "x")
        pf1 = af.pushforward_ir(ir)
        pf2 = af.pushforward_ir(pf1)
        pf3 = af.pushforward_ir(pf2)
        pf4 = af.pushforward_ir(pf3)
        level0 = "a"
        level1 = "b"
        level2 = ("c", "d")
        level3 = (("e", "f"), ("g", "h"))
        level4 = ((("i", "j"), ("k", "l")), (("m", "n"), ("o", "p")))
        result = af.run_ir(pf4, ((((level0, level1), level2), level3), level4))
        expected = (
            ((("[a]", "[b]"), ("[c]", "[d]")), (("[e]", "[f]"), ("[g]", "[h]"))),
            ((("[i]", "[j]"), ("[k]", "[l]")), (("[m]", "[n]"), ("[o]", "[p]"))),
        )
        assert result == expected


class TestTriplePullback:
    def test_triple_pullback(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        pb1 = af.pullback_ir(ir)
        pb2 = af.pullback_ir(pb1)
        pb3 = af.pullback_ir(pb2)
        p = "a"
        c1 = "g1"
        c2 = ("g2_p", "g2_c")
        c3 = (("g3_pp", "g3_pc"), ("g3_cp", "g3_cc"))
        result = af.run_ir(pb3, (((p, c1), c2), c3))
        (((out_p, out_c1), _), _) = result
        assert out_p == "a!"
        assert out_c1 == "g1"


class TestMixedDeepNesting:
    def test_batch_pushforward_pullback(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        pb = af.pullback_ir(ir)
        pf = af.pushforward_ir(pb)
        b = af.batch_ir(pf, in_axes=(list, list))
        p_primals = ["a", "b"]
        p_cotangents = ["g1", "g2"]
        t_primals = ["ta", "tb"]
        t_cotangents = ["tg1", "tg2"]
        result = af.run_ir(b, ((p_primals, p_cotangents), (t_primals, t_cotangents)))
        assert len(result) == 2

    def test_pushforward_batch_pullback(self):
        def f(x):
            return af.format("[{}]", x)

        ir = af.build_ir(f, "x")
        pb = af.pullback_ir(ir)
        b = af.batch_ir(pb, in_axes=(list, list))
        pf = af.pushforward_ir(b)
        p_primals = ["a", "b"]
        p_cotangents = ["g1", "g2"]
        t_primals = ["ta", "tb"]
        t_cotangents = ["tg1", "tg2"]
        result = af.run_ir(pf, ((p_primals, p_cotangents), (t_primals, t_cotangents)))
        (p_out, t_out) = result
        assert p_out == (["[a]", "[b]"], ["g1", "g2"])
        assert t_out == (["[ta]", "[tb]"], ["tg1", "tg2"])

    def test_pullback_pushforward_batch(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        b = af.batch_ir(ir)
        pf = af.pushforward_ir(b)
        pb = af.pullback_ir(pf)
        p_inputs = ["a", "b"]
        t_inputs = ["ta", "tb"]
        cotangent_out = (["g1", "g2"], ["tg1", "tg2"])
        result = af.run_ir(pb, ((p_inputs, t_inputs), cotangent_out))
        (primal_result, cotangent_result) = result
        assert primal_result == (["a!", "b!"], ["ta!", "tb!"])

    def test_batch_batch_pushforward(self):
        def f(x):
            return af.format("<{}>", x)

        ir = af.build_ir(f, "x")
        pf = af.pushforward_ir(ir)
        b1 = af.batch_ir(pf, in_axes=(list, list))
        b2 = af.batch_ir(b1, in_axes=(list, list))
        primals = [["a", "b"], ["c"]]
        tangents = [["ta", "tb"], ["tc"]]
        result = af.run_ir(b2, primals, tangents)
        assert result == (
            [["<a>", "<b>"], ["<c>"]],
            [["<ta>", "<tb>"], ["<tc>"]],
        )

    def test_pushforward_pushforward_batch(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        b = af.batch_ir(ir)
        pf1 = af.pushforward_ir(b)
        pf2 = af.pushforward_ir(pf1)
        p_xs = ["a", "b"]
        t1_xs = ["t1a", "t1b"]
        t2 = (["t2pa", "t2pb"], ["t2ta", "t2tb"])
        result = af.run_ir(pf2, ((p_xs, t1_xs), t2))
        assert result == (
            (["a!", "b!"], ["t1a!", "t1b!"]),
            (["t2pa!", "t2pb!"], ["t2ta!", "t2tb!"]),
        )


class TestAlternatingTransforms:
    def test_pf_pb_pf_pb(self):
        def f(x):
            return af.format("[{}]", x)

        ir = af.build_ir(f, "x")
        pb1 = af.pullback_ir(ir)
        pf1 = af.pushforward_ir(pb1)
        pb2 = af.pullback_ir(pf1)
        pf2 = af.pushforward_ir(pb2)
        p = "x"
        c1 = "g"
        t1 = ("tp", "tc")
        c2 = (("cpp", "cpc"), ("ctp", "ctc"))
        t2 = (
            (
                (("tppp", "tppc"), ("tpcp", "tpcc")),
                (("tpcpp", "tpcpc"), ("tpccp", "tpccc")),
            ),
            (
                (("tcpp", "tcpc"), ("tccp", "tccc")),
                (("tccpp", "tccpc"), ("tcccp", "tcccc")),
            ),
        )
        result = af.run_ir(pf2, (((((p, c1), t1), c2), t2)))
        assert result is not None

    def test_batch_pf_batch_pf(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        pf1 = af.pushforward_ir(ir)
        b1 = af.batch_ir(pf1, in_axes=(list, list))
        pf2 = af.pushforward_ir(b1)
        b2 = af.batch_ir(pf2, in_axes=(list, list))
        input_tree = (
            (
                [["a", "b"], ["c", "d"]],
                [["ta", "tb"], ["tc", "td"]],
            ),
            (
                [["qa", "qb"], ["qc", "qd"]],
                [["qta", "qtb"], ["qtc", "qtd"]],
            ),
        )
        result = af.run_ir(b2, input_tree)
        assert result == (
            ([["a!", "b!"], ["c!", "d!"]], [["ta!", "tb!"], ["tc!", "td!"]]),
            ([["qa!", "qb!"], ["qc!", "qd!"]], [["qta!", "qtb!"], ["qtc!", "qtd!"]]),
        )


class TestDeepWithMultipleArgs:
    def test_double_batch_two_args(self):
        def f(a, b):
            return af.concat(a, b)

        ir = af.build_ir(f, "a", "b")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        a_vals = [["a1", "a2"], ["a3"]]
        b_vals = [["b1", "b2"], ["b3"]]
        result = af.run_ir(b2, a_vals, b_vals)
        assert result == [["a1b1", "a2b2"], ["a3b3"]]

    def test_pushforward_batch_two_args(self):
        def f(a, b):
            return af.format("{}-{}", a, b)

        ir = af.build_ir(f, "a", "b")
        b = af.batch_ir(ir)
        pf = af.pushforward_ir(b)
        p_a = ["a1", "a2"]
        p_b = ["b1", "b2"]
        t_a = ["ta1", "ta2"]
        t_b = ["tb1", "tb2"]
        result = af.run_ir(pf, ((p_a, p_b), (t_a, t_b)))
        assert result == (
            ["a1-b1", "a2-b2"],
            ["ta1-tb1", "ta2-tb2"],
        )

    def test_pullback_double_batch_two_args(self):
        def f(a, b):
            return af.concat(a, b)

        ir = af.build_ir(f, "a", "b")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        pb = af.pullback_ir(b2)
        a_vals = [["a1", "a2"], ["a3"]]
        b_vals = [["b1", "b2"], ["b3"]]
        cotangent = [["g1", "g2"], ["g3"]]
        result = af.run_ir(pb, ((a_vals, b_vals), cotangent))
        (primal_out, cotangent_in) = result
        assert primal_out == [["a1b1", "a2b2"], ["a3b3"]]


class TestEdgeCasesDeepNesting:
    def test_empty_at_deepest_level(self):
        def f(x):
            return af.format("{}!", x)

        ir = af.build_ir(f, "x")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        b3 = af.batch_ir(b2)
        inputs = [[[], []], []]
        result = af.run_ir(b3, inputs)
        assert result == [[[], []], []]

    def test_single_element_deep(self):
        def f(x):
            return af.concat(x, "!")

        ir = af.build_ir(f, "x")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        b3 = af.batch_ir(b2)
        pf = af.pushforward_ir(b3)
        primals = [[["a"]]]
        tangents = [[["t"]]]
        result = af.run_ir(pf, (primals, tangents))
        assert result == ([[["a!"]]], [[["t!"]]])

    def test_mixed_empty_nonempty(self):
        def f(x):
            return af.format("<{}>", x)

        ir = af.build_ir(f, "x")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        inputs = [["a", "b"], [], ["c"]]
        result = af.run_ir(b2, inputs)
        assert result == [["<a>", "<b>"], [], ["<c>"]]


class TestChainedOperations:
    def test_format_concat_deep_batch(self):
        def f(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        ir = af.build_ir(f, "x")
        b1 = af.batch_ir(ir)
        b2 = af.batch_ir(b1)
        pf = af.pushforward_ir(b2)
        primals = [["a", "b"], ["c"]]
        tangents = [["ta", "tb"], ["tc"]]
        result = af.run_ir(pf, (primals, tangents))
        assert result == (
            [["[a]!", "[b]!"], ["[c]!"]],
            [["[ta]!", "[tb]!"], ["[tc]!"]],
        )

    def test_multi_step_all_transforms(self):
        def f(x):
            a = af.format("({}", x)
            b = af.concat(a, ")")
            return b

        ir = af.build_ir(f, "x")
        pb = af.pullback_ir(ir)
        b = af.batch_ir(pb, in_axes=(list, list))
        pf = af.pushforward_ir(b)
        b2 = af.batch_ir(pf, in_axes=(list, list))
        p_p = [["a", "b"]]
        p_c = [["g1", "g2"]]
        t_p = [["ta", "tb"]]
        t_c = [["tg1", "tg2"]]
        result = af.run_ir(b2, ((p_p, p_c), (t_p, t_c)))
        assert result is not None
