"""
Convex Environment — Base Case.

Simple convex function: compute(x) = 2*x + 3.
6 test cases, ~36 chars of code.
"""

import ast
import re
from typing import Dict, Any

from core.env_base import AbstractEnvironment


class ConvexEnvironment(AbstractEnvironment):
    """Simple convex function environment (Base Case)."""

    def __init__(self, obfuscation_level: int = 0, max_steps: int = 20):
        ground_truth = """def compute(x):
    return 2 * x + 3"""
        super().__init__(
            ground_truth_code=ground_truth,
            max_steps=max_steps,
            obfuscation_level=obfuscation_level,
        )
        self.test_cases = [
            (0, 3), (1, 5), (2, 7), (-1, 1), (10, 23), (5, 13),
        ]

    def _get_task_description(self) -> str:
        return (
            "Implement a function compute(x) that returns 2*x + 3.\n"
            "Examples: compute(0)=3, compute(1)=5, compute(2)=7, compute(-1)=1"
        )

    def _evaluate(self, code: str) -> Dict[str, Any]:
        passed = 0
        total = len(self.test_cases)
        errors = []
        has_syntax_error = False

        try:
            ast.parse(code)
            namespace = {}
            exec(code, namespace)

            if "compute" not in namespace:
                errors.append("Function 'compute' not found")
                return {
                    "loss": 1.0,
                    "passed_tests": 0,
                    "total_tests": total,
                    "errors": errors,
                    "has_syntax_error": True,
                }

            compute_func = namespace["compute"]
            for x, expected in self.test_cases:
                try:
                    result = compute_func(x)
                    if result == expected:
                        passed += 1
                    else:
                        errors.append(f"Test failed: compute({x}) = {result}, expected {expected}")
                except Exception as e:
                    errors.append(f"Runtime error on input {x}: {str(e)}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {str(e)}")
            has_syntax_error = True

        loss = 1.0 - (passed / total) if total > 0 else 1.0
        return {
            "loss": loss,
            "passed_tests": passed,
            "total_tests": total,
            "errors": errors,
            "has_syntax_error": has_syntax_error,
        }
