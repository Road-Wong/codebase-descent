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
    """
    
    def __init__(self, level: int = 1):
        """
        Initialize the semantic stripper.
        
        Args:
            level: Obfuscation level (0-3)
        """
        self.level = level
        self.var_mapping: Dict[str, str] = {}
        self.func_mapping: Dict[str, str] = {}
        self.var_counter = 0
        self.func_counter = 0
        
        # Reserved Python keywords and builtins to not rename
        self.reserved = {
            'True', 'False', 'None', 'and', 'or', 'not', 'is', 'in',
            'if', 'else', 'elif', 'for', 'while', 'break', 'continue',
            'def', 'class', 'return', 'yield', 'import', 'from', 'as',
            'try', 'except', 'finally', 'raise', 'with', 'lambda',
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
            'set', 'tuple', 'bool', 'sum', 'max', 'min', 'abs', 'sorted',
            'enumerate', 'zip', 'map', 'filter', 'any', 'all'
        }
    
    def _get_new_var_name(self, original_name: str) -> str:
        """Generate obfuscated variable name based on level."""
        if original_name in self.reserved:
            return original_name
        
        if original_name not in self.var_mapping:
            self.var_counter += 1
            if self.level == 1:
                # Level 1: Simple v_1, v_2, ... (readable but meaningless)
                self.var_mapping[original_name] = f"v_{self.var_counter}"
            elif self.level >= 2:
                # Level 2+: Visually confusing Il1I, IlI1, ... (hard to distinguish)
                # Use combinations of I, l, 1 to create confusion
                chars = ['I', 'l', '1']
                name = 'Il'
                num = self.var_counter
                while num > 0:
                    name += chars[num % 3]
                    num //= 3
                self.var_mapping[original_name] = name
        
        return self.var_mapping[original_name]
    
    def _get_new_func_name(self, original_name: str) -> str:
        """Generate obfuscated function name."""
        if original_name in self.reserved or original_name.startswith('__'):
            return original_name
        
        if original_name not in self.func_mapping:
            self.func_counter += 1
            if self.level == 1:
                self.func_mapping[original_name] = f"f_{self.func_counter}"
            elif self.level >= 2:
                self.func_mapping[original_name] = f"fn{self.func_counter}Il"
        
        return self.func_mapping[original_name]
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename function definitions and remove docstrings."""
        # Rename function
        if self.level >= 1:
            node.name = self._get_new_func_name(node.name)
        
        # Remove docstring (Level 3+)
        if self.level >= 3 and node.body:
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]  # Remove docstring
        
        # Continue visiting child nodes
        self.generic_visit(node)
        return node
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variable references."""
        if self.level >= 1:
            node.id = self._get_new_var_name(node.id)
        return node
    
    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Rename function arguments."""
        if self.level >= 1:
            node.arg = self._get_new_var_name(node.arg)
        return node
    
    def obfuscate(self, code_str: str) -> Tuple[str, Dict[str, str]]:
        """
        Obfuscate the given code string.
        
        Args:
            code_str: Original Python code
            
        Returns:
            Tuple of (obfuscated_code, mapping_dict)
        """
        if self.level == 0:
            return code_str, {}
        
        try:
            # Parse the code into AST
            tree = ast.parse(code_str)
            
            # Apply transformations
            self.visit(tree)
            
            # Convert back to code
            obfuscated = ast.unparse(tree)
            
            # Additional cleanup for Level 3+
            if self.level >= 3:
                # Remove inline comments
                obfuscated = re.sub(r'#.*$', '', obfuscated, flags=re.MULTILINE)
                # Remove excessive blank lines
                obfuscated = re.sub(r'\n\n+', '\n\n', obfuscated)
            
            # Combine mappings for reference
            all_mappings = {**self.var_mapping, **self.func_mapping}
            
            return obfuscated.strip(), all_mappings
            
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")


def obfuscate_code(code: str, level: int = 1) -> Tuple[str, Dict[str, str]]:
    """
    Convenience function to obfuscate code.
    
    Args:
        code: Original Python code
        level: Obfuscation level (0-3)
        
    Returns:
        Tuple of (obfuscated_code, mapping_dict)
        
    Example:
        >>> code = '''
        ... def calculate_sum(numbers):
        ...     total = 0
        ...     for num in numbers:
        ...         total += num
        ...     return total
        ... '''
        >>> obfuscated, mapping = obfuscate_code(code, level=1)
        >>> print(obfuscated)
        def f_1(v_1):
            v_2 = 0
            for v_3 in v_1:
                v_2 += v_3
            return v_2
    """
    stripper = SemanticStripper(level=level)
    return stripper.obfuscate(code)


def create_obfuscode_benchmark(original_code: str, test_cases: list, level: int = 1) -> dict:
    """
    Create an ObfusCode benchmark from original code and test cases.
    
    Args:
        original_code: Clean Python code with semantic names
        test_cases: List of (input, expected_output) tuples
        level: Obfuscation level
        
    Returns:
        Dictionary containing obfuscated code, test cases, and metadata
    """
    obfuscated, mapping = obfuscate_code(original_code, level)
    
    return {
        'original_code': original_code,
        'obfuscated_code': obfuscated,
        'mapping': mapping,
        'test_cases': test_cases,
        'level': level,
        'difficulty_multiplier': 1.0 + (level * 0.5)  # Estimated difficulty increase
    }


if __name__ == '__main__':
    # Demo: Show the effect of different obfuscation levels
    sample_code = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    previous = 0
    current = 1
    for i in range(2, n + 1):
        next_value = previous + current
        previous = current
        current = next_value
    return current
'''
    
    print("=" * 70)
    print("OBFUSCATION DEMO")
    print("=" * 70)
    
    for level in [0, 1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"Level {level}:")
        print('='*70)
        obfuscated, mapping = obfuscate_code(sample_code, level)
        print(obfuscated)
        if mapping:
            print(f"\nMappings: {list(mapping.items())[:5]}...")  # Show first 5 mappings
