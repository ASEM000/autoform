import threading
import functools as ft
from collections.abc import Callable
import autoform.core as core


class TestInterpreterRuleMapping:
    def test_basic_get_set(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        p = core.Primitive("test_basic")

        @ft.partial(mapping.set, p)
        def rule(x):
            return x

        assert mapping.get(p) is rule

    def test_duplicate_raises(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        p = core.Primitive("test_duplicate")

        @ft.partial(mapping.set, p)
        def rule1(x):
            return x

        try:

            @ft.partial(mapping.set, p)
            def rule2(x):
                return x

            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_contains(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        p1 = core.Primitive("test_contains_1")
        p2 = core.Primitive("test_contains_2")

        @ft.partial(mapping.set, p1)
        def rule(x):
            return x

        assert p1 in mapping
        assert p2 not in mapping

    def test_iter(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        p1 = core.Primitive("test_iter_1")
        p2 = core.Primitive("test_iter_2")

        @ft.partial(mapping.set, p1)
        def rule1(x):
            return x

        @ft.partial(mapping.set, p2)
        def rule2(x):
            return x

        items = list(mapping)
        prims = [p for p, _ in items]
        assert p1 in prims
        assert p2 in prims

    def test_concurrent_registration(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        results = []
        errors = []

        def register_rule(thread_id):
            try:
                p = core.Primitive(f"concurrent_{thread_id}")

                @ft.partial(mapping.set, p)
                def rule(x):
                    return x * thread_id

                results.append((thread_id, p))
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=register_rule, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 50
        for thread_id, p in results:
            rule = mapping.get(p)
            assert rule(1) == thread_id

    def test_reentrant_lock(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        p = core.Primitive("reentrant")

        @ft.partial(mapping.set, p)
        def rule(x):
            return x

        with mapping.lock:
            with mapping.lock:
                assert p in mapping

    def test_iteration_during_contains(self):
        mapping = core.InterpreterRuleMapping[Callable]()
        prims = [core.Primitive(f"iter_contains_{i}") for i in range(10)]
        for p in prims:

            @ft.partial(mapping.set, p)
            def rule(x, prim=p):
                return x

        errors = []

        def iterate():
            try:
                for _ in range(100):
                    list(mapping)
            except Exception as e:
                errors.append(e)

        def check_contains():
            try:
                for _ in range(100):
                    for p in prims:
                        _ = p in mapping
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=iterate),
            threading.Thread(target=iterate),
            threading.Thread(target=check_contains),
            threading.Thread(target=check_contains),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
