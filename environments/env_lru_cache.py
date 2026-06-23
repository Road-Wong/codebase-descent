"""
LRU Cache Environment — Stateful data structure task.

8 test cases covering: basic ops, capacity=1, update existing key,
get updates recency, larger capacity, multiple updates, empty gets.
"""

from typing import Dict, Any

from core.env_base import AbstractEnvironment


class LRUCacheEnvironment(AbstractEnvironment):
    """Environment for implementing an LRU Cache."""

    def __init__(self, obfuscation_level: int = 0, max_steps: int = 20):
        ground_truth = """class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
        self.cache[key] = value
        self.order.append(key)
"""
        super().__init__(
            ground_truth_code=ground_truth,
            max_steps=max_steps,
            obfuscation_level=obfuscation_level,
        )
        self.test_cases = [
            {"capacity": 2, "operations": [("put",(1,1)),("put",(2,2)),("get",(1,)),("put",(3,3)),("get",(2,)),("put",(4,4)),("get",(1,)),("get",(3,)),("get",(4,))], "expected": [None,None,1,None,-1,None,-1,3,4]},
            {"capacity": 1, "operations": [("put",(2,1)),("get",(2,)),("put",(3,2)),("get",(2,)),("get",(3,))], "expected": [None,1,None,-1,2]},
            {"capacity": 2, "operations": [("put",(1,1)),("put",(2,2)),("get",(1,)),("put",(1,10)),("get",(1,)),("put",(3,3)),("get",(2,))], "expected": [None,None,1,None,10,None,-1]},
            {"capacity": 2, "operations": [("put",(1,1)),("put",(2,2)),("get",(1,)),("put",(3,3)),("get",(1,)),("get",(2,))], "expected": [None,None,1,None,1,-1]},
            {"capacity": 3, "operations": [("put",(1,1)),("put",(2,2)),("put",(3,3)),("put",(4,4)),("get",(1,)),("get",(2,)),("get",(3,)),("get",(4,))], "expected": [None,None,None,None,-1,2,3,4]},
            {"capacity": 2, "operations": [("put",(1,1)),("put",(1,2)),("put",(1,3)),("get",(1,)),("put",(2,2)),("get",(1,))], "expected": [None,None,None,3,None,3]},
            {"capacity": 2, "operations": [("get",(1,)),("put",(1,1)),("get",(1,)),("get",(2,))], "expected": [-1,None,1,-1]},
            {"capacity": 3, "operations": [("put",(1,1)),("put",(2,2)),("put",(3,3)),("get",(1,)),("put",(4,4)),("get",(2,)),("put",(5,5)),("get",(1,)),("get",(3,)),("get",(4,))], "expected": [None,None,None,1,None,-1,None,1,-1,4]},
        ]

    def _get_task_description(self) -> str:
        return (
            "Implement an LRUCache class with:\n"
            "- __init__(capacity): Initialize with given capacity\n"
            "- get(key): Return value if exists, else -1 (marks as recently used)\n"
            "- put(key, value): Insert or update, evict LRU if at capacity"
        )

    def _evaluate(self, code: str) -> Dict[str, Any]:
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        has_syntax_error = False

        try:
            namespace = {}
            exec(code, namespace)

            if "LRUCache" not in namespace:
                errors.append("LRUCache class not found")
                return {
                    "loss": 1.0, "passed_tests": 0, "total_tests": total_tests,
                    "errors": errors, "has_syntax_error": True,
                }

            LRUCache = namespace["LRUCache"]
            for i, test_case in enumerate(self.test_cases):
                try:
                    cache = LRUCache(test_case["capacity"])
                    results = []
                    for op, args in test_case["operations"]:
                        if op == "get":
                            results.append(cache.get(args[0]))
                        elif op == "put":
                            results.append(cache.put(args[0], args[1]))
                    if results == test_case["expected"]:
                        passed_tests += 1
                    else:
                        errors.append(f"Test {i+1} failed: expected {test_case['expected']}, got {results}")
                except Exception as e:
                    errors.append(f"Test {i+1} error: {str(e)}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {str(e)}")
            has_syntax_error = True

        loss = 1.0 - (passed_tests / total_tests) if total_tests > 0 else 1.0
        return {
            "loss": loss, "passed_tests": passed_tests, "total_tests": total_tests,
            "errors": errors[:3], "has_syntax_error": has_syntax_error,
        }
