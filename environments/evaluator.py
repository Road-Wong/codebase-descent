"""
SGD-Style Evaluator — Random Single-Sample Feedback

This module enforces the core "physical law" that prevents LLM cheating:
the Agent receives feedback from exactly ONE randomly-chosen failing test
per evaluation step, not a global error report.

Why this matters:
  In standard batch evaluation (all 26 tests at once), the LLM receives
  a rich gradient signal that lets it "jump" to the solution in one step.
  This is like giving SGD the full-batch gradient — it converges trivially.

  By contrast, single-sample feedback forces the Agent to:
  1. Fix ONE problem at a time (local step)
  2. Re-discover other failures on subsequent steps (exploration)
  3. Deal with noisy/conflicting signals (stochastic gradient)

  The random shuffle ensures the Agent cannot learn a fixed order of
  fixes — each step is an independent random draw from the test distribution.

Analogy to SGD:
  - Full test suite  →  Full-batch gradient (deterministic, easy)
  - Single test      →  Single-sample gradient (stochastic, hard)
  - Random shuffle   →  Random sampling from data distribution
  - Stop at 1st fail →  Truncated evaluation (early stopping in SGD)
"""

from __future__ import annotations

import ast
import random
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple


class SGDEvaluator:
    """
    Evaluates code against a hidden test suite using SGD-style feedback.

    The Agent NEVER sees:
      - The test case source code
      - The expected outputs
      - How many tests exist
      - Which test was selected

    The Agent ONLY sees:
      - A single error message (if any test failed)
      - The scalar loss (passed/total, but not the denominator)
    """

    def __init__(
        self,
        test_cases: List[Tuple],
        entry_point: str,
        eval_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            test_cases: List of (input, expected_output) tuples.
                        These are NEVER exposed to the Agent.
            entry_point: The function/class name to call (e.g. "SimpleInterpreter").
            eval_fn: Custom eval function(code, test_case) -> (passed, error_msg).
                     If None, uses the default exec-based evaluator.
            seed: Random seed for reproducibility.
        """
        self._test_cases = test_cases
        self._entry_point = entry_point
        self._eval_fn = eval_fn or self._default_eval
        self._rng = random.Random(seed)

    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        Evaluate code using SGD-style single-sample feedback.

        Returns:
            {
                "loss": float,           # 1 - passed/total (for metrics)
                "passed_tests": int,     # total passed (hidden from agent)
                "total_tests": int,      # total tests (hidden from agent)
                "error_msg": str,        # THE SINGLE ERROR the agent sees
                "has_syntax_error": bool,
            }

        The Agent should ONLY see error_msg.  The loss, passed_tests, and
        total_tests are for internal metrics/logging only.
        """
        # Check syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "loss": 1.0,
                "passed_tests": 0,
                "total_tests": len(self._test_cases),
                "error_msg": f"SyntaxError: {e}",
                "has_syntax_error": True,
            }

        # Shuffle test cases — each evaluation is a random permutation
        shuffled = list(self._test_cases)
        self._rng.shuffle(shuffled)

        passed = 0
        first_error: Optional[str] = None

        for test_case in shuffled:
            ok, err_msg = self._eval_fn(code, test_case)
            if ok:
                passed += 1
            else:
                # STOP at first failure — this is the SGD single-sample rule
                if first_error is None:
                    first_error = err_msg
                break  # <-- KEY: stop immediately, don't run remaining tests

        total = len(self._test_cases)

        # If ALL tests passed, there is no error
        error_msg = first_error if first_error else ""

        return {
            "loss": 1.0 - (passed / total) if total > 0 else 1.0,
            "passed_tests": passed,
            "total_tests": total,
            "error_msg": error_msg,
            "has_syntax_error": False,
        }

    def evaluate_full(self, code: str) -> Dict[str, Any]:
        """
        Full-batch evaluation (for internal metrics ONLY).

        This runs ALL tests and returns the true loss.
        NEVER call this in the Agent's feedback loop — it defeats
        the purpose of SGD single-sample feedback.

        Use only for:
          - Final scoring after episode ends
          - Logging the true loss trajectory
          - Computing oscillation metrics
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "loss": 1.0, "passed_tests": 0, "total_tests": len(self._test_cases),
                "errors": [f"SyntaxError: {e}"], "has_syntax_error": True,
            }

        passed = 0
        errors = []
        for test_case in self._test_cases:
            ok, err_msg = self._eval_fn(code, test_case)
            if ok:
                passed += 1
            else:
                errors.append(err_msg)

        total = len(self._test_cases)
        return {
            "loss": 1.0 - (passed / total) if total > 0 else 1.0,
            "passed_tests": passed,
            "total_tests": total,
            "errors": errors,
            "has_syntax_error": False,
        }

    def _default_eval(self, code: str, test_case: Tuple) -> Tuple[bool, str]:
        """
        Default evaluation: exec the code and call the entry point.

        Returns:
            (passed: bool, error_message: str)
        """
        try:
            namespace = {}
            exec(code, namespace)

            if self._entry_point not in namespace:
                return False, f"'{self._entry_point}' not defined in code"

            cls_or_fn = namespace[self._entry_point]

            # Handle class-based entry points (SimpleInterpreter, LRUCache, etc.)
            if isinstance(cls_or_fn, type):
                instance = cls_or_fn()
                return self._eval_class_instance(instance, test_case)
            else:
                # Function-based
                args, expected = test_case
                if not isinstance(args, tuple):
                    args = (args,)
                result = cls_or_fn(*args)
                if result == expected:
                    return True, ""
                return False, f"Output mismatch: expected {expected}, got {result}"

        except Exception as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            # Return a clean single-line error (no stack trace leakage)
            return False, f"RuntimeError: {type(e).__name__}: {e}"

    def _eval_class_instance(
        self, instance: Any, test_case: Tuple
    ) -> Tuple[bool, str]:
        """Evaluate a class instance against a test case."""
        # Test case format depends on the environment
        # For SimpleInterpreter: (expr, expected_result)
        if isinstance(test_case, tuple) and len(test_case) == 2:
            expr, expected = test_case
            if isinstance(expr, str):
                try:
                    result = instance.evaluate(expr)
                    if result == expected:
                        return True, ""
                    return False, (
                        f"Output mismatch on evaluate('{expr}'): "
                        f"expected {expected}, got {result}"
                    )
                except Exception as e:
                    return False, f"RuntimeError on evaluate('{expr}'): {e}"

        # Fallback: try calling with test_case as args
        try:
            if isinstance(test_case, tuple):
                result = instance(*test_case)
            else:
                result = instance(test_case)
            return (True, "") if result else (False, f"Returned falsy: {result}")
        except Exception as e:
            return False, f"RuntimeError: {e}"
