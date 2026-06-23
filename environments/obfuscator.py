"""
Semantic Stripper - ObfusCode Generator

This module implements AST-based code obfuscation to create "de-semanticized" benchmarks.
By removing semantic cues (variable names, docstrings), we force LLMs to rely on pure
logical reasoning rather than pattern matching.

This is the core of our experimental setup - it allows us to control the "ruggedness"
of the loss landscape by adjusting the obfuscation level.
"""

import ast
import re
from typing import Dict, Tuple, Optional


class SemanticStripper(ast.NodeTransformer):
    """
    AST-based code obfuscator that removes semantic information.

    Obfuscation Levels:
    - Level 0: Clean code (baseline)
    - Level 1: Rename variables to v_1, v_2, ... (structure preserved, semantics removed)
    - Level 2: Aggressive renaming with visually confusing names (IllI, IlIl, ...)
    - Level 3: Level 2 + remove all comments and docstrings
      Public API method names (evaluate, tokenize, etc.) are PRESERVED so
      the test harness can still call them — the agent must fix the
      internal logic, not reconstruct the API from scratch.
    """

    # Methods whose names must be preserved at all obfuscation levels,
    # because the test harness (or public API) calls them by name.
    PRESERVE_METHODS = frozenset({
        'evaluate', 'tokenize', '__init__',
        '_eval_expr', '_eval_term', '_eval_factor',
    })

    def __init__(self, level: int = 1):
        self.level = level
        self.var_mapping: Dict[str, str] = {}
        self.func_mapping: Dict[str, str] = {}
        self.var_counter = 0
        self.func_counter = 0
        self.reserved = {
            'True', 'False', 'None', 'and', 'or', 'not', 'is', 'in',
            'if', 'else', 'elif', 'for', 'while', 'break', 'continue',
            'def', 'class', 'return', 'yield', 'import', 'from', 'as',
            'try', 'except', 'finally', 'raise', 'with', 'lambda',
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
            'set', 'tuple', 'bool', 'sum', 'max', 'min', 'abs', 'sorted',
            'enumerate', 'zip', 'map', 'filter', 'any', 'all',
            'self', 'cls',  # Python class method keywords — never rename
        }

    def _get_new_var_name(self, original_name: str) -> str:
        if original_name in self.reserved:
            return original_name
        if original_name not in self.var_mapping:
            self.var_counter += 1
            if self.level == 1:
                self.var_mapping[original_name] = f"v_{self.var_counter}"
            elif self.level >= 2:
                chars = ['I', 'l', '1']
                name = 'Il'
                num = self.var_counter
                while num > 0:
                    name += chars[num % 3]
                    num //= 3
                self.var_mapping[original_name] = name
        return self.var_mapping[original_name]

    def _get_new_func_name(self, original_name: str) -> str:
        if original_name in self.reserved or original_name.startswith('__'):
            return original_name
        # Preserve public API methods so test harness can still call them
        if original_name in self.PRESERVE_METHODS:
            return original_name
        if original_name not in self.func_mapping:
            self.func_counter += 1
            if self.level == 1:
                self.func_mapping[original_name] = f"f_{self.func_counter}"
            elif self.level >= 2:
                self.func_mapping[original_name] = f"fn{self.func_counter}Il"
        return self.func_mapping[original_name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if self.level >= 1:
            node.name = self._get_new_func_name(node.name)
        if self.level >= 3 and node.body:
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if self.level >= 1:
            node.id = self._get_new_var_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if self.level >= 1:
            node.arg = self._get_new_var_name(node.arg)
        return node

    def obfuscate(self, code_str: str) -> Tuple[str, Dict[str, str]]:
        if self.level == 0:
            return code_str, {}
        try:
            tree = ast.parse(code_str)
            self.visit(tree)
            obfuscated = ast.unparse(tree)
            if self.level >= 3:
                obfuscated = re.sub(r'#.*$', '', obfuscated, flags=re.MULTILINE)
                obfuscated = re.sub(r'\n\n+', '\n\n', obfuscated)
            all_mappings = {**self.var_mapping, **self.func_mapping}
            return obfuscated.strip(), all_mappings
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")


def obfuscate_code(code: str, level: int = 1) -> Tuple[str, Dict[str, str]]:
    """Convenience function to obfuscate code."""
    stripper = SemanticStripper(level=level)
    return stripper.obfuscate(code)


def create_obfuscode_benchmark(original_code: str, test_cases: list, level: int = 1) -> dict:
    """Create an ObfusCode benchmark from original code and test cases."""
    obfuscated, mapping = obfuscate_code(original_code, level)
    return {
        'original_code': original_code,
        'obfuscated_code': obfuscated,
        'mapping': mapping,
        'test_cases': test_cases,
        'level': level,
        'difficulty_multiplier': 1.0 + (level * 0.5)
    }
