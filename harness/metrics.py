import ast
import re
from typing import Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_edit_distance(code1: str, code2: str) -> float:
    """
    Calculate normalized edit distance between code snippets.
    
    Args:
        code1: First code snippet
        code2: Second code snippet
        
    Returns:
        Normalized distance in [0, 1]
    """
    # Normalize whitespace
    c1 = re.sub(r'\s+', ' ', code1.strip())
    c2 = re.sub(r'\s+', ' ', code2.strip())
    
    distance = levenshtein_distance(c1, c2)
    max_len = max(len(c1), len(c2))
    
    if max_len == 0:
        return 0.0
    
    return distance / max_len


def token_level_distance(code1: str, code2: str) -> Tuple[float, int, int]:
    """
    Calculate token-level edit distance.
    
    Args:
        code1: First code snippet
        code2: Second code snippet
        
    Returns:
        Tuple of (normalized_distance, num_edits, total_tokens)
    """
    # Simple tokenization (split by whitespace and operators)
    def tokenize(code):
        # Split by whitespace and common operators
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [t for t in tokens if t.strip()]
    
    tokens1 = tokenize(code1)
    tokens2 = tokenize(code2)
    
    distance = levenshtein_distance(' '.join(tokens1), ' '.join(tokens2))
    total_tokens = max(len(tokens1), len(tokens2))
    
    if total_tokens == 0:
        return 0.0, 0, 0
    
    return distance / total_tokens, distance, total_tokens


def code_similarity(code1: str, code2: str) -> float:
    """
    Calculate similarity score between code snippets.

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Similarity score in [0, 1], where 1 is identical
    """
    distance = normalized_edit_distance(code1, code2)
    return 1.0 - distance


# ---------------------------------------------------------------------------
# AST-based distance — structural code comparison
# ---------------------------------------------------------------------------

def _ast_to_canonical(code: str) -> str:
    """
    Parse code into AST and dump it in a canonical string form.

    This strips whitespace, comments, and formatting — only structure remains.
    Two codes with identical logic but different formatting produce the same string.
    """
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        # If code doesn't parse, fall back to raw string
        return code


def ast_distance(code1: str, code2: str) -> float:
    """
    Normalized AST edit distance between two code snippets.

    Computes the Levenshtein distance between the canonical AST dump
    strings, normalized to [0, 1].

    This measures *structural* divergence:
      - Renaming a variable         → small distance (AST structure unchanged)
      - Changing operator precedence → large distance (AST tree shape changes)
      - Adding/removing a branch    → large distance

    Use case: Quantify "oscillation amplitude" in the code manifold.
    If the Agent keeps flipping between two structurally different codes,
    the AST distance between consecutive steps will be high.

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Normalized distance in [0, 1], where 0 = structurally identical.
    """
    ast1 = _ast_to_canonical(code1)
    ast2 = _ast_to_canonical(code2)

    if ast1 == ast2:
        return 0.0

    dist = levenshtein_distance(ast1, ast2)
    max_len = max(len(ast1), len(ast2))
    if max_len == 0:
        return 0.0

    return dist / max_len


# ---------------------------------------------------------------------------
# Composite metric — combines test loss and structural distance
# ---------------------------------------------------------------------------

class CompositeMetric:
    """
    Composite loss function for code optimization:

        L_total = L_test + λ * L_ast

    Where:
        L_test: Fraction of failed tests (computed by evaluator).
                This is the TRUE loss — used for convergence check.
                The Agent only sees ONE error message, not this number.

        L_ast:  Normalized AST edit distance from initial code.
                Measures how far the Agent has wandered structurally.
                Used to detect oscillation (high L_ast = wild swings).

        λ:      Weight for structural penalty (default 0.2).

    The composite metric is recorded at each step for post-hoc analysis
    (trajectory plots, phase transition detection).  The Agent itself
    only receives the single-sample SGD error message.
    """

    def __init__(self, initial_code: str, lambda_ast: float = 0.2):
        self._initial_code = initial_code
        self._lambda = lambda_ast
        self._history: list = []

    def compute(
        self,
        current_code: str,
        test_loss: float,
        step: int = 0,
    ) -> dict:
        """
        Compute the composite metric.

        Args:
            current_code: The current code snapshot.
            test_loss: L_test from the evaluator (1 - passed/total).
            step: Current optimization step.

        Returns:
            {
                "L_test": float,
                "L_ast": float,
                "L_total": float,
                "step": int,
            }
        """
        l_ast = ast_distance(self._initial_code, current_code)
        l_total = test_loss + self._lambda * l_ast

        result = {
            "L_test": test_loss,
            "L_ast": l_ast,
            "L_total": l_total,
            "step": step,
        }
        self._history.append(result)
        return result

    @property
    def history(self) -> list:
        """Full metric history for trajectory analysis."""
        return self._history

    def oscillation_index(self) -> float:
        """
        Compute oscillation index from AST distance history.

        OI = (total path length) / (net displacement) - 1

        Where:
          - Total path length = sum of consecutive AST distances
          - Net displacement  = AST distance from first to last code

        High OI → Agent is going in circles (limit cycle)
        OI = 0  → Agent moved in a straight line (ideal convergence)
        """
        if len(self._history) < 2:
            return 0.0

        # Total path: sum of L_ast at each step
        total_path = sum(h["L_ast"] for h in self._history[1:])

        # Net displacement: L_ast of final vs initial
        net_disp = self._history[-1]["L_ast"]

        if net_disp == 0:
            return 10.0 if total_path > 0 else 0.0  # returned to start

        return max(0.0, (total_path / net_disp) - 1.0)
