"""
Tests for code obfuscator (merged from test_framework.py and test_new_architecture.py).
"""

from environments import obfuscate_code


def test_obfuscator_levels():
    code = "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total"

    for level in [0, 1, 2]:
        obfuscated, mapping = obfuscate_code(code, level=level)
        assert isinstance(obfuscated, str)
        if level > 0:
            assert len(mapping) > 0
