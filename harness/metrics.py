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
