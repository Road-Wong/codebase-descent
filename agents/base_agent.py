"""
Diff-Only Agent Base — Constrains Action Space to Local Patches

This module enforces the critical "step size" constraint:
the Agent CANNOT emit full code replacements.  It MUST emit
SEARCH/REPLACE blocks that modify only a localized region.

Why this matters:
  If the Agent can output arbitrary full code, it can "teleport"
  across the loss landscape — writing the ground truth from scratch
  in a single step.  This is not optimization; it's memorization.

  By restricting to SEARCH/REPLACE:
  1. Each step modifies only a small region (bounded step size)
  2. The Agent must identify WHAT to change (credit assignment)
  3. Failed patches count as wasted steps (exploration cost)
  4. The Agent accumulates a trajectory of local fixes (gradient path)

Analogy to optimization:
  Full code output   →  Newton step (global, exact, cheating)
  SEARCH/REPLACE     →  Gradient step (local, approximate, honest)
  Failed patch       →  Step rejected by line search (wasted iteration)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.agent_base import AbstractAgent
from core.types import Action, ActionFormat, Observation
from core.protocol import (
    apply_patch,
    make_action,
    validate_action,
    ProtocolError,
    PatchResult,
)


class DiffAgent(AbstractAgent):
    """
    Base class for agents that output SEARCH/REPLACE patches.

    Subclasses implement `_generate_patch()` which returns a raw
    SEARCH/REPLACE string.  This base class:
      1. Validates the format (ProtocolError if malformed)
      2. Tests the patch against current code (reports if SEARCH not found)
      3. Wraps result in a properly-typed Action

    The Agent's `act()` method never touches the environment directly.
    It only produces a patch; the environment decides whether to apply it.
    """

    def __init__(self):
        self._step_count = 0
        self._patch_history: List[Dict[str, Any]] = []
        self._error_feedback: Optional[str] = None

    def act(self, observation: Observation) -> Action:
        """
        Produce a SEARCH/REPLACE Action from the current Observation.

        Pipeline:
          1. Generate raw patch text (subclass logic, possibly LLM)
          2. Validate format
          3. Preview: check if SEARCH block matches current code
          4. Return Action with format=SEARCH_REPLACE
        """
        self._step_count += 1

        # Generate the raw patch text
        raw_patch = self._generate_patch(observation)

        # Validate format
        is_valid, err = validate_action(raw_patch)
        if not is_valid:
            # Record the failure
            self._patch_history.append({
                "step": self._step_count,
                "success": False,
                "error": f"ProtocolError: {err}",
                "patch": raw_patch,
            })
            # Return the malformed action anyway — the environment will
            # penalize it.  This is analogous to a rejected line search step.
            return Action(
                patch=raw_patch,
                format=ActionFormat.SEARCH_REPLACE,
                confidence=0.0,
                reasoning=f"Format error: {err}",
            )

        # Preview: check if patch would apply
        preview = apply_patch(observation.current_code, raw_patch)
        if not preview.success:
            # SEARCH block not found — record and return with low confidence
            self._patch_history.append({
                "step": self._step_count,
                "success": False,
                "error": "; ".join(preview.errors),
                "patch": raw_patch,
            })
            return Action(
                patch=raw_patch,
                format=ActionFormat.SEARCH_REPLACE,
                confidence=0.1,
                reasoning=f"Preview warning: {'; '.join(preview.errors)}",
            )

        # Valid patch that would apply
        self._patch_history.append({
            "step": self._step_count,
            "success": True,
            "patch": raw_patch,
        })
        return Action(
            patch=raw_patch,
            format=ActionFormat.SEARCH_REPLACE,
            confidence=0.8,
        )

    def reset(self) -> None:
        """Reset agent state between episodes."""
        self._step_count = 0
        self._patch_history = []
        self._error_feedback = None

    def get_statistics(self) -> Dict[str, Any]:
        """Return patch success statistics."""
        total = len(self._patch_history)
        success = sum(1 for p in self._patch_history if p["success"])
        return {
            "total_steps": self._step_count,
            "valid_patches": success,
            "invalid_patches": total - success,
            "patch_success_rate": success / total if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _generate_patch(self, observation: Observation) -> str:
        """
        Generate a raw SEARCH/REPLACE patch string.

        Subclasses must implement this.  The returned string MUST
        follow the protocol format:
            <<<< SEARCH
            [old code]
            ====
            [new code]
            >>>>

        Args:
            observation: Current observation (code, error, loss, etc.)

        Returns:
            Raw SEARCH/REPLACE text.
        """
        raise NotImplementedError("Subclasses must implement _generate_patch()")


class RandomDiffAgent(DiffAgent):
    """
    Debug/test agent that makes random small modifications.

    Useful for testing the environment without an LLM.
    Tries to fix the first line of code that contains an error keyword.
    """

    def __init__(self):
        super().__init__()

    def _generate_patch(self, observation: Observation) -> str:
        """Generate a trivial patch based on the error message."""
        code = observation.current_code
        error = observation.errors[0] if observation.errors else ""

        # If there's an error, try to fix the first obvious issue
        if "not defined" in error or "not found" in error:
            # Missing class/function — add a stub
            if "SimpleInterpreter" in error:
                return make_action(
                    "",
                    "class SimpleInterpreter:\n    def __init__(self):\n        pass\n",
                )

        # If syntax error, try to fix common issues
        if observation.has_syntax_error:
            if "def " in code and ":\n" not in code:
                # Missing colon
                for line in code.split("\n"):
                    if "def " in line and not line.rstrip().endswith(":"):
                        return make_action(
                            line,
                            line.rstrip() + ":",
                        )

        # Default: return a no-op patch (search for first line, replace with itself)
        lines = code.strip().split("\n")
        if lines:
            return make_action(lines[0], lines[0])

        return make_action("", "")
