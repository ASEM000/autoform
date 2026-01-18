import functools as ft

import pytest

import autoform as af

greet_p = af.core.Primitive("greet")


def greet(name: str, *, greeting: str = "Hello", punctuation: str = "!") -> str:
    return greet_p.bind((name, dict(greeting=greeting, punctuation=punctuation)))


@ft.partial(af.core.impl_rules.set, greet_p)
def impl_greet(in_tree) -> str:
    match in_tree:
        case (name, {"greeting": greeting, "punctuation": punctuation}):
            return f"{greeting}, {name}{punctuation}"


@ft.partial(af.core.eval_rules.set, greet_p)
def eval_greet(in_tree) -> af.core.Var:
    return af.core.Var(str)


@ft.partial(af.core.push_rules.set, greet_p)
def pushforward_greet(in_tree):
    primals, tangents = in_tree
    return impl_greet(primals), impl_greet(tangents)


@ft.partial(af.core.pull_fwd_rules.set, greet_p)
def pullback_fwd_greet(in_tree):
    out = impl_greet(in_tree)
    residuals = in_tree
    return out, residuals


@ft.partial(af.core.pull_bwd_rules.set, greet_p)
def pullback_bwd_greet(residuals, out_cotangent):
    match residuals:
        case (_, {"greeting": greeting, "punctuation": punct}):
            return (
                out_cotangent,
                {
                    "greeting": af.ad.zero_cotangent(greeting),
                    "punctuation": af.ad.zero_cotangent(punct),
                },
            )


class TestKwargsBuildIR:
    def test_kwargs_in_ir(self):
        def program(name):
            return greet(name, greeting="Hi", punctuation="?")

        ir = af.trace(program)("World")
        assert len(ir.ireqns) == 1
        eqn = ir.ireqns[0]
        assert eqn.prim == greet_p
        match eqn.in_irtree:
            case (name, {"greeting": greeting, "punctuation": punctuation}):
                assert isinstance(name, af.core.IRVar)
                assert isinstance(greeting, af.core.IRLit)
                assert isinstance(punctuation, af.core.IRLit)
            case _:
                pytest.fail("Unexpected in_irtree structure")

    def test_kwargs_execution(self):
        def program(name):
            return greet(name, greeting="Hi", punctuation="?")

        ir = af.trace(program)("World")
        result = af.call(ir)("World")
        assert result == "Hi, World?"


class TestKwargsPushforward:
    def test_pushforward_with_kwargs_ir(self):
        test_p = af.core.Primitive("test_kwargs_pf")

        @ft.partial(af.core.eval_rules.set, test_p)
        def eval_rule(in_tree):
            return af.core.Var(str)

        @ft.partial(af.core.impl_rules.set, test_p)
        def impl_rule(in_tree):
            x, kwargs = in_tree
            repeat = kwargs["repeat"]
            return x * repeat

        @ft.partial(af.core.push_rules.set, test_p)
        def pf_rule(in_tree):
            primals, tangents = in_tree
            return impl_rule(primals), impl_rule(tangents)

        def program(x, *, repeat=1):
            return test_p.bind((x, dict(repeat=repeat)))

        ir = af.trace(program)("A", repeat=3)
        match ir.in_irtree:
            case (x, {"repeat": repeat}):
                assert isinstance(x, af.core.IRVar)
                assert isinstance(repeat, af.core.IRLit)


class TestKwargsPullback:
    def test_pullback_greet_ir_structure(self):
        def program(name):
            return greet(name, greeting="Hi", punctuation="?")

        ir = af.trace(program)("World")
        pb_ir = af.pullback(ir)
        assert len(pb_ir.in_irtree) == 2
        assert len(pb_ir.out_irtree) == 2
