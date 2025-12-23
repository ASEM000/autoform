import threading
import functools as ft
from collections.abc import Callable
import autoform.core as core


class TestInterpreterRuleMapping:
    def test_basic_get_set(self):
        mapping = core.InterpreterRuleMapping[Callable](override=False)
        p = core.Primitive("test_basic")

        @ft.partial(mapping.set, p)
        def rule(x):
            return x

        assert mapping.get(p) is rule

    def test_override_false_raises(self):
        mapping = core.InterpreterRuleMapping[Callable](override=False)
        p = core.Primitive("test_override")

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

    def test_override_true_allows(self):
        mapping = core.InterpreterRuleMapping[Callable](override=True)
        p = core.Primitive("test_override_true")

        @ft.partial(mapping.set, p)
        def rule1(x):
            return x

        @ft.partial(mapping.set, p)
        def rule2(x):
            return x * 2

        assert mapping.get(p) is rule2

    def test_contains(self):
        mapping = core.InterpreterRuleMapping[Callable](override=False)
        p1 = core.Primitive("test_contains_1")
        p2 = core.Primitive("test_contains_2")

        @ft.partial(mapping.set, p1)
        def rule(x):
            return x

        assert p1 in mapping
        assert p2 not in mapping

    def test_iter(self):
        mapping = core.InterpreterRuleMapping[Callable](override=False)
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
        mapping = core.InterpreterRuleMapping[Callable](override=True)
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

    def test_concurrent_read_write(self):
        mapping = core.InterpreterRuleMapping[Callable](override=True)
        p = core.Primitive("concurrent_rw")
        read_results = []
        errors = []

        @ft.partial(mapping.set, p)
        def initial_rule(x):
            return x

        def reader(reader_id):
            try:
                for _ in range(100):
                    rule = mapping.get(p)
                    if rule is not None:
                        read_results.append(reader_id)
            except Exception as e:
                errors.append((reader_id, e))

        def writer(writer_id):
            try:
                for i in range(10):

                    @ft.partial(mapping.set, p)
                    def rule(x, w=writer_id, n=i):
                        return x * w * n

            except Exception as e:
                errors.append((writer_id, e))

        readers = [threading.Thread(target=reader, args=(i,)) for i in range(10)]
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()
        assert len(errors) == 0

    def test_reentrant_lock(self):
        mapping = core.InterpreterRuleMapping[Callable](override=False)
        p = core.Primitive("reentrant")

        @ft.partial(mapping.set, p)
        def rule(x):
            return x

        with mapping.lock:
            with mapping.lock:
                assert p in mapping

    def test_iteration_during_contains(self):
        mapping = core.InterpreterRuleMapping[Callable](override=False)
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
