import functools as ft
import threading

import autoform as af


class TestInterpreterRuleMapping:
    def test_basic_get_set(self):
        mapping = af.core.InterpreterRuleMapping()
        p = af.core.Primitive("test_basic")

        @ft.partial(mapping.set, p)
        def rule(x):
            return x

        assert mapping.get(p) is rule

    def test_duplicate_raises(self):
        mapping = af.core.InterpreterRuleMapping()
        p = af.core.Primitive("test_duplicate")

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

    def test_concurrent_registration(self):
        mapping = af.core.InterpreterRuleMapping()
        results = []
        errors = []

        def set_rule(thread_id):
            try:
                p = af.core.Primitive(f"concurrent_{thread_id}")

                @ft.partial(mapping.set, p)
                def rule(x):
                    return x * thread_id

                results.append((thread_id, p))
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=set_rule, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 50
        for thread_id, p in results:
            rule = mapping.get(p)
            assert rule(1) == thread_id
