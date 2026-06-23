"""
Search/Replace Action Protocol

Implements the Aider-style SEARCH/REPLACE format for incremental code patches.
This is the "gradient step" Δc that the Agent produces:

    <<<< SEARCH
    [old code block]
    ====
    [new code block]
    >>>>

The protocol enforces:
1. Structured format — Agent cannot emit arbitrary text
2. Minimal diff — only the changed region is specified
3. Fuzzy matching — tolerates minor whitespace differences
4. Precise errors — when a patch fails, the Agent knows exactly why
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


class ProtocolError(Exception):
    """Raised when an action violates the SEARCH/REPLACE protocol."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass
class SearchReplaceBlock:
    """A single SEARCH → REPLACE transformation."""
    search: str
    replace: str
    line_number: int  # line in the action text where this block starts


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    code: str  # resulting code (original if failed)
    applied: int  # number of blocks successfully applied
    errors: List[str]  # error messages for failed blocks


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_SEARCH_MARKER = "<<<< SEARCH"
_SEPARATOR_MARKER = "===="
_REPLACE_MARKER = ">>>>"


def parse_action(action_text: str) -> List[SearchReplaceBlock]:
    """
    Parse an action text into a list of SEARCH/REPLACE blocks.

    Format:
        <<<< SEARCH
        [old code]
        ====
        [new code]
        >>>>

    Multiple blocks can appear in a single action.

    Raises:
        ProtocolError: If the format is invalid.
    """
    blocks: List[SearchReplaceBlock] = []
    lines = action_text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == _SEARCH_MARKER:
            block, end_idx = _parse_single_block(lines, i)
            blocks.append(block)
            i = end_idx
        elif line == "":
            i += 1  # skip blank lines between blocks
        else:
            raise ProtocolError(
                f"Expected '{_SEARCH_MARKER}' at line {i + 1}, got: {line!r}",
                hint="Each block must start with <<<< SEARCH on its own line.",
            )

    if not blocks:
        raise ProtocolError(
            "No SEARCH/REPLACE blocks found in action.",
            hint="Action must contain at least one <<<< SEARCH ... ==== ... >>>> block.",
        )

    return blocks


def _parse_single_block(lines: List[str], start: int) -> Tuple[SearchReplaceBlock, int]:
    """Parse one SEARCH/REPLACE block starting at `start`."""
    search_lines: List[str] = []
    replace_lines: List[str] = []
    state = "search"  # "search" | "replace"
    i = start + 1  # skip <<<< SEARCH

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == _SEPARATOR_MARKER and state == "search":
            state = "replace"
            i += 1
            continue
        elif stripped == _REPLACE_MARKER:
            # End of block
            return SearchReplaceBlock(
                search="\n".join(search_lines),
                replace="\n".join(replace_lines),
                line_number=start + 1,
            ), i + 1

        if state == "search":
            search_lines.append(line)
        else:
            replace_lines.append(line)

        i += 1

    raise ProtocolError(
        f"Unterminated SEARCH/REPLACE block starting at line {start + 1}. "
        f"Missing '{_REPLACE_MARKER}' marker.",
        hint="Every <<<< SEARCH block must end with >>>>",
    )


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

def _normalize_for_comparison(text: str) -> str:
    """Normalize whitespace for fuzzy comparison."""
    # Strip trailing whitespace per line, normalize line endings
    lines = text.split("\n")
    return "\n".join(line.rstrip() for line in lines)


def find_search_block(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    """
    Find the SEARCH block in the source code.

    Returns (start_index, end_index) in the haystack string, or None if not found.

    Matching strategy (in order of preference):
    1. Exact match
    2. Normalized match (strip trailing whitespace)
    3. Fuzzy match (difflib, threshold 0.8)
    """
    if not needle.strip():
        return (0, 0)  # empty search matches beginning

    # 1. Exact match
    idx = haystack.find(needle)
    if idx != -1:
        return (idx, idx + len(needle))

    # 2. Normalized match
    norm_haystack = _normalize_for_comparison(haystack)
    norm_needle = _normalize_for_comparison(needle)
    idx = norm_haystack.find(norm_needle)
    if idx != -1:
        # Map back to original indices
        # We need to find where this normalized substring starts in the original
        # Approximate: use the same index
        return (idx, idx + len(needle))

    # 3. Fuzzy match — find best substring match
    best_ratio = 0.0
    best_start = -1
    best_end = -1

    # Slide a window of needle length across haystack
    needle_len = len(needle)
    # Allow window to be +/- 30% of needle length
    min_window = max(1, int(needle_len * 0.7))
    max_window = int(needle_len * 1.3) + 1

    for window_size in range(min_window, min(max_window, len(haystack) + 1)):
        for start in range(len(haystack) - window_size + 1):
            candidate = haystack[start:start + window_size]
            ratio = difflib.SequenceMatcher(None, needle, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = start + window_size

    if best_ratio >= 0.8:
        return (best_start, best_end)

    return None


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------

def apply_patch(original_code: str, action_text: str) -> PatchResult:
    """
    Apply a SEARCH/REPLACE action to the original code.

    Args:
        original_code: The current code to patch.
        action_text: The action text containing SEARCH/REPLACE blocks.

    Returns:
        PatchResult with success status, resulting code, and diagnostics.

    Raises:
        ProtocolError: If the action format is invalid.
    """
    blocks = parse_action(action_text)

    code = original_code
    applied = 0
    errors: List[str] = []

    for block in blocks:
        loc = find_search_block(code, block.search)
        if loc is None:
            # Provide helpful error with context
            snippet = block.search[:80].replace("\n", "\\n")
            errors.append(
                f"Block at line {block.line_number}: "
                f"SEARCH text not found in code. "
                f"Searched for: {snippet!r}"
            )
            continue

        start, end = loc
        code = code[:start] + block.replace + code[end:]
        applied += 1

    return PatchResult(
        success=(applied == len(blocks)),
        code=code,
        applied=applied,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def make_action(search: str, replace: str) -> str:
    """
    Construct an action text from search and replace strings.

    Args:
        search: The code to find.
        replace: The code to replace it with.

    Returns:
        Formatted action text.
    """
    return (
        f"{_SEARCH_MARKER}\n"
        f"{search}\n"
        f"{_SEPARATOR_MARKER}\n"
        f"{replace}\n"
        f"{_REPLACE_MARKER}"
    )


def validate_action(action_text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an action text without applying it.

    Returns:
        (is_valid, error_message)
    """
    try:
        parse_action(action_text)
        return True, None
    except ProtocolError as e:
        return False, str(e)


def extract_protocol_block(text: str) -> str:
    """
    Extract SEARCH/REPLACE protocol block from text that may contain
    explanation or preamble before the markers.

    If the text contains <<<< SEARCH, extract from the first marker
    to the last >>>>. Otherwise return the original text unchanged.
    """
    idx = text.find("<<<< SEARCH")
    if idx == -1:
        return text

    # Find the last >>>> marker
    last_end = text.rfind(">>>>")
    if last_end == -1:
        return text[idx:]

    return text[idx:last_end + 4]
