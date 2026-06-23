"""
Diff/Patch utilities for gradient-step semantics.

The Agent's Action contains a unified diff (Δc), analogous to the
negative gradient step in continuous optimization:

    c_{t+1} = c_t + Δc

This module provides:
- generate_diff(old, new) → unified diff string
- apply_diff(old, diff) → new code (or raise PatchError)
- DiffAction: Action subclass that carries both diff and target code
"""

from __future__ import annotations

import difflib
from typing import Optional, Tuple

from .types import Action, ActionFormat


class PatchError(Exception):
    """Raised when a patch cannot be applied."""
    pass


def generate_diff(old_code: str, new_code: str, filename: str = "code.py") -> str:
    """
    Generate a unified diff from old_code to new_code.

    Args:
        old_code: Original code (c_t)
        new_code: Modified code (c_{t+1})
        filename: Filename for diff header

    Returns:
        Unified diff string
    """
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "\n".join(diff)


def apply_diff(old_code: str, diff_text: str) -> str:
    """
    Apply a unified diff to old_code.

    This is a simplified patcher that handles standard unified diff format.

    Args:
        old_code: Current code to patch
        diff_text: Unified diff text

    Returns:
        New code after applying the patch

    Raises:
        PatchError: If the patch cannot be applied
    """
    if not diff_text.strip():
        return old_code

    lines = diff_text.splitlines()

    # Parse hunks
    hunks = []
    current_hunk = None

    for line in lines:
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            try:
                parts = line.split("@@")
                ranges = parts[1].strip() if len(parts) > 1 else ""
                old_range, new_range = ranges.split(" ")
                old_start = int(old_range.split(",")[0].lstrip("-")) - 1
                new_start = int(new_range.split(",")[0].lstrip("+")) - 1
                current_hunk = {
                    "old_start": old_start,
                    "new_start": new_start,
                    "lines": [],
                }
            except (ValueError, IndexError):
                raise PatchError(f"Malformed hunk header: {line}")
        elif current_hunk is not None:
            if line.startswith("-"):
                current_hunk["lines"].append(("remove", line[1:]))
            elif line.startswith("+"):
                current_hunk["lines"].append(("add", line[1:]))
            elif line.startswith(" "):
                current_hunk["lines"].append(("context", line[1:]))
            elif line.startswith("\\"):
                # "\ No newline at end of file"
                pass

    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        return old_code

    # Apply hunks in reverse order (so line numbers don't shift)
    old_lines = old_code.splitlines()
    result_lines = list(old_lines)

    for hunk in reversed(hunks):
        old_start = hunk["old_start"]

        # Build the replacement: context lines stay, remove lines are replaced by add lines
        new_region = []
        for op, line in hunk["lines"]:
            if op == "context":
                new_region.append(line)
            elif op == "add":
                new_region.append(line)
            # "remove" lines are skipped (they existed in old, not in new)

        # Count how many old lines this hunk covers
        old_count = sum(1 for op, _ in hunk["lines"] if op in ("context", "remove"))
        end = old_start + old_count
        result_lines[old_start:end] = new_region

    return "\n".join(result_lines)


def make_diff_action(
    old_code: str,
    new_code: str,
    filename: str = "code.py",
    confidence: float = 0.5,
    reasoning: Optional[str] = None,
) -> Action:
    """
    Create an Action with unified diff from old_code to new_code.

    The patch field contains the unified diff.
    The metadata field carries the target new_code for reliable application.

    Args:
        old_code: Current code (c_t)
        new_code: New code (c_{t+1})
        filename: Filename for diff header
        confidence: Agent's confidence in this action
        reasoning: Optional chain-of-thought

    Returns:
        Action with UNIFIED_DIFF format
    """
    diff = generate_diff(old_code, new_code, filename)

    # If diff is empty (code unchanged), create a minimal diff
    if not diff.strip():
        diff = generate_diff(old_code, new_code, filename)

    return Action(
        patch=diff,
        format=ActionFormat.UNIFIED_DIFF,
        confidence=confidence,
        reasoning=reasoning,
        # Store target code in metadata for reliable application
        metadata={"_target_code": new_code},
    )


def extract_target_code(action: Action, fallback_code: str) -> str:
    """
    Extract the target code from an Action.

    Prefers metadata._target_code if available (reliable).
    Falls back to applying the diff (may fail on complex patches).

    Args:
        action: The Action to extract from
        fallback_code: Current code (for diff application fallback)

    Returns:
        The target code after applying the action
    """
    # Check for target code in metadata
    if hasattr(action, "_target_code") and action._target_code:
        return action._target_code

    # Check metadata dict
    if hasattr(action, "metadata") and action.metadata:
        if "_target_code" in action.metadata:
            return action.metadata["_target_code"]

    # Apply diff
    if action.format == ActionFormat.FULL_CODE:
        return action.patch

    if action.format == ActionFormat.UNIFIED_DIFF:
        try:
            return apply_diff(fallback_code, action.patch)
        except PatchError:
            return fallback_code

    return action.patch
