"""
Abstract RL Environment for Code Optimization.

This module defines the strict Gymnasium-style interface that all
environments must implement.  The key contract:

    observation, reward, done, truncated, info = env.step(action)

The Agent never sees the environment's internal State — it only
receives Observations.  This enforces the information asymmetry
that makes the optimization dynamics meaningful.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .types import Action, ActionFormat, Observation, State


class AbstractEnvironment(ABC):
    """
    Strict RL interface for code optimization environments.

    Analogy to control theory:
        Plant:  c_{t+1} = f(c_t, Δc_t)     # code update
        Loss:   L(c_t)                       # evaluation
        Obs:    o_t = O(s_t)                 # partial observability

    Subclasses must implement:
        - _build_observation(): convert internal State → Observation
        - _apply_patch(): apply Action.patch to current code
        - _evaluate(): run test suite against code
        - _get_task_description(): return the task spec for the Agent

    The step() method is final — do not override it.  It orchestrates:
        1. Apply patch → new code
        2. Evaluate → loss, test results
        3. Build observation
        4. Check termination
    """

    def __init__(
        self,
        ground_truth_code: str,
        max_steps: int = 20,
        obfuscation_level: int = 0,
    ):
        self._ground_truth = ground_truth_code
        self._max_steps = max_steps
        self._obfuscation_level = obfuscation_level

        # Mutable state
        self._state = State(code="", step=0, loss=1.0)
        self._done = False
        self._task_description: str = ""

    # ------------------------------------------------------------------
    # Public API (final — do not override)
    # ------------------------------------------------------------------

    def reset(
        self,
        initial_code: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Reset the environment to initial state.

        Args:
            initial_code: Starting code.  If None, uses ground truth
                          (optionally obfuscated).
            seed: Random seed for reproducibility.

        Returns:
            Initial Observation.
        """
        if initial_code is None:
            initial_code = self._prepare_initial_code()

        self._state = State(code=initial_code, step=0, loss=1.0)
        self._done = False
        self._task_description = self._get_task_description()

        # Evaluate initial state
        eval_result = self._evaluate(initial_code)
        self._state.loss = eval_result["loss"]

        return self._build_observation(self._state, eval_result)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Execute one optimization step.

        This is the core RL transition:
            s_t --[action: Δc]--> s_{t+1}

        Args:
            action: Agent's patch (Δc).

        Returns:
            (observation, reward, done, truncated, info)

            - observation: what the Agent sees next
            - reward: -loss (RL convention: higher = better)
            - done: True if loss == 0 (solved)
            - truncated: True if max_steps reached
            - info: extra diagnostics
        """
        if self._done:
            raise RuntimeError("Episode finished. Call reset() first.")

        # 1. Apply patch → new code
        try:
            new_code = self._apply_patch(self._state.code, action)
        except Exception as e:
            # Patch application failed — penalize, don't crash
            new_code = self._state.code  # keep old code
            info = {"patch_error": str(e), "action_format": action.format}
            obs = self._build_observation(self._state, {
                "loss": self._state.loss,
                "passed_tests": 0,
                "total_tests": 0,
                "errors": [f"Patch failed: {e}"],
                "has_syntax_error": True,
            })
            return obs, -self._state.loss, False, False, info

        # 2. Evaluate → loss, test results
        eval_result = self._evaluate(new_code)

        # 3. Update state
        self._state = State(
            code=new_code,
            step=self._state.step + 1,
            loss=eval_result["loss"],
            metadata=eval_result,
        )

        # 4. Build observation
        obs = self._build_observation(self._state, eval_result)

        # 5. Check termination
        done = eval_result["loss"] == 0.0  # solved
        truncated = self._state.step >= self._max_steps  # out of steps

        if done:
            self._done = True

        reward = -eval_result["loss"]  # RL convention: higher = better

        info = {
            "passed_tests": eval_result.get("passed_tests", 0),
            "total_tests": eval_result.get("total_tests", 0),
            "errors": eval_result.get("errors", []),
            "step": self._state.step,
        }

        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _evaluate(self, code: str) -> Dict[str, Any]:
        """
        Run the test suite against code.

        Must return a dict with at least:
            - loss: float ∈ [0, 1]
            - passed_tests: int
            - total_tests: int
            - errors: list[str]
            - has_syntax_error: bool
        """
        ...

    @abstractmethod
    def _get_task_description(self) -> str:
        """Return the task description that the Agent sees."""
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def _apply_patch(self, current_code: str, action: Action) -> str:
        """
        Apply the agent's patch to the current code.

        Supports:
        - SEARCH_REPLACE: Aider-style SEARCH/REPLACE protocol (preferred)
        - UNIFIED_DIFF: apply diff, with metadata._target_code fallback
        - FULL_CODE: direct replacement (legacy)
        """
        if action.format == ActionFormat.SEARCH_REPLACE:
            from .protocol import apply_patch
            result = apply_patch(current_code, action.patch)
            if not result.errors:
                return result.code
            else:
                raise ValueError(f"Patch errors: {'; '.join(result.errors)}")

        from .diff_utils import extract_target_code
        return extract_target_code(action, current_code)

    def _prepare_initial_code(self) -> str:
        """
        Prepare the initial code for the episode.

        Default: returns ground truth (possibly obfuscated).
        Override to inject noise via NoiseInjector.
        """
        if self._obfuscation_level > 0:
            from environments.obfuscator import obfuscate_code
            code, _ = obfuscate_code(
                self._ground_truth, level=self._obfuscation_level
            )
            return code
        return self._ground_truth

    def _build_observation(
        self, state: State, eval_result: Dict[str, Any]
    ) -> Observation:
        """
        Convert internal State → Observation (what the Agent sees).

        Default implementation.  Override to add SMO-specific fields
        (momentum, gradients) or to redact information.
        """
        return Observation(
            current_code=state.code,
            task_description=self._task_description,
            loss=eval_result.get("loss", state.loss),
            passed_tests=eval_result.get("passed_tests", 0),
            total_tests=eval_result.get("total_tests", 0),
            errors=eval_result.get("errors", []),
            has_syntax_error=eval_result.get("has_syntax_error", False),
            step=state.step,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Current internal state (for logging, not for Agent)."""
        return self._state

    @property
    def ground_truth(self) -> str:
        """Ground truth code (c*).  Never exposed to Agent."""
        return self._ground_truth

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def is_done(self) -> bool:
        return self._done


# ---------------------------------------------------------------------------
# Concrete Environment: loads from environments/assets/
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "environments" / "assets"


class CodeEnv(AbstractEnvironment):
    """
    Concrete environment that loads ground truth from environments/assets/.

    On reset():
      1. Loads original_code.py from assets
      2. Applies obfuscation (if configured)
      3. Sets the degraded code as initial state

    The Agent then tries to recover the original code through
    SEARCH/REPLACE patches.
    """

    def __init__(
        self,
        asset_name: str = "original_code",
        obfuscation_level: int = 0,
        max_steps: int = 20,
    ):
        self._asset_name = asset_name
        ground_truth = self._load_asset(asset_name)

        # Saddle task test suite
        self._test_cases = [
            ("1+1", 2), ("2*3", 6), ("10-5", 5), ("8/2", 4),
            ("2+3*4", 14), ("(2+3)*4", 20), ("2*3+4*5", 26), ("5+3*2-4", 7),
            ("10-2-3", 5), ("1+2+3+4", 10), ("100/10/2", 5), ("(1+2)*(3+4)", 21),
            ("((1+2)*3)+4", 13), ("(1+(2*3))+4", 11), ("((2+3)*(4+5))", 45),
            ("1+2*3+4*5", 27), ("(1+2)*(3+4)*(5+6)", 231), ("10-5+3*2", 11),
            ("100/10+5*2", 20), ("0+0", 0), ("1*1", 1), ("10/10", 1),
            ("5-5", 0), ("100+200", 300), ("50*4", 200), ("1000/10", 100),
        ]

        super().__init__(
            ground_truth_code=ground_truth,
            max_steps=max_steps,
            obfuscation_level=obfuscation_level,
        )

    @staticmethod
    def _load_asset(name: str) -> str:
        """Load a Python file from environments/assets/."""
        path = _ASSETS_DIR / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {path}")
        return path.read_text(encoding="utf-8")

    def _get_task_description(self) -> str:
        return (
            "Implement a SimpleInterpreter class with methods:\n"
            "- tokenize(expr): tokenize arithmetic expression into (type, value) pairs\n"
            "- evaluate(expr): evaluate expression and return integer result\n"
            "- _eval_expr, _eval_term, _eval_factor: recursive descent parser helpers\n"
            "Support +, -, *, /, parentheses. Integer division for /. Operator precedence applies.\n"
            "Examples: '2+3*4'=14, '(2+3)*4'=20, '1+1'=2, '10/3'=3"
        )

    def _evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate code against the 26-expression test suite."""
        import ast

        passed = 0
        total = len(self._test_cases)
        errors = []
        has_syntax_error = False

        try:
            ast.parse(code)
            namespace = {}
            exec(code, namespace)

            if "SimpleInterpreter" not in namespace:
                errors.append("Class 'SimpleInterpreter' not found")
                return {
                    "loss": 1.0, "passed_tests": 0, "total_tests": total,
                    "errors": errors, "has_syntax_error": True,
                }

            interpreter = namespace["SimpleInterpreter"]()
            for expr, expected in self._test_cases:
                try:
                    result = interpreter.evaluate(expr)
                    if result == expected:
                        passed += 1
                    else:
                        errors.append(f"evaluate('{expr}') = {result}, expected {expected}")
                except Exception as e:
                    errors.append(f"Runtime error on '{expr}': {e}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {e}")
            has_syntax_error = True

        loss = 1.0 - (passed / total) if total > 0 else 1.0
        return {
            "loss": loss, "passed_tests": passed, "total_tests": total,
            "errors": errors, "has_syntax_error": has_syntax_error,
        }

    def _prepare_initial_code(self) -> str:
        """
        Load from asset, then apply obfuscation.

        This is called by reset() when no initial_code is provided.
        """
        raw = self._load_asset(self._asset_name)

        if self._obfuscation_level > 0:
            from environments.obfuscator import obfuscate_code
            code, _ = obfuscate_code(raw, level=self._obfuscation_level)
            return code

        return raw
