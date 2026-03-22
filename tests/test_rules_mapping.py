# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft
import threading

import autoform as af


class TestInterpreterRuleMapping:
    def test_basic_get_set(self):
        mapping = af.core.InterpreterRuleMapping()
        p = af.core.Prim("test_basic")

        @ft.partial(mapping.set, p)
        def rule(x):
            return x

        assert mapping.get(p) is rule

    def test_duplicate_raises(self):
        mapping = af.core.InterpreterRuleMapping()
        p = af.core.Prim("test_duplicate")

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
                p = af.core.Prim(f"concurrent_{thread_id}")

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
