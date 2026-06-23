"""
Saddle Environment — Main Case.

Task: Implement a simple interpreter with coupled Lexer and Parser.
26 test cases covering operator precedence, nesting, edge cases.
"""

import ast
import re
from typing import Dict, Any

from core.env_base import AbstractEnvironment


class SaddleEnvironment(AbstractEnvironment):
    """
    Complex coupled system environment (Main Case).
    Task: Implement a simple interpreter with coupled Lexer and Parser.
    Modifying one component affects the other (high coupling).
    """

    def __init__(self, obfuscation_level: int = 0, max_steps: int = 20):
        ground_truth = """class SimpleInterpreter:
    def __init__(self):
        self.tokens = []

    def tokenize(self, expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit():
                num = ''
                while i < len(expr) and expr[i].isdigit():
                    num += expr[i]
                    i += 1
                tokens.append(('NUM', int(num)))
            elif expr[i] in '+-*/':
                tokens.append(('OP', expr[i]))
                i += 1
            elif expr[i] in '()':
                tokens.append(('PAREN', expr[i]))
                i += 1
            else:
                i += 1
        return tokens

    def evaluate(self, expr):
        tokens = self.tokenize(expr)
        return self._eval_expr(tokens, 0)[0]

    def _eval_expr(self, tokens, pos):
        left, pos = self._eval_term(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == 'OP' and tokens[pos][1] in '+-':
            op = tokens[pos][1]
            pos += 1
            right, pos = self._eval_term(tokens, pos)
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left, pos

    def _eval_term(self, tokens, pos):
        left, pos = self._eval_factor(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == 'OP' and tokens[pos][1] in '*/':
            op = tokens[pos][1]
            pos += 1
            right, pos = self._eval_factor(tokens, pos)
            if op == '*':
                left = left * right
            else:
                left = left // right
        return left, pos

    def _eval_factor(self, tokens, pos):
        if tokens[pos][0] == 'NUM':
            return tokens[pos][1], pos + 1
        elif tokens[pos][0] == 'PAREN' and tokens[pos][1] == '(':
            pos += 1
            result, pos = self._eval_expr(tokens, pos)
            pos += 1  # skip ')'
            return result, pos
        return 0, pos
"""
        super().__init__(
            ground_truth_code=ground_truth,
            max_steps=max_steps,
            obfuscation_level=obfuscation_level,
        )
        self.test_cases = [
            ("1+1", 2), ("2*3", 6), ("10-5", 5), ("8/2", 4),
            ("2+3*4", 14), ("(2+3)*4", 20), ("2*3+4*5", 26), ("5+3*2-4", 7),
            ("10-2-3", 5), ("1+2+3+4", 10), ("100/10/2", 5), ("(1+2)*(3+4)", 21),
            ("((1+2)*3)+4", 13), ("(1+(2*3))+4", 11), ("((2+3)*(4+5))", 45),
            ("1+2*3+4*5", 27), ("(1+2)*(3+4)*(5+6)", 231), ("10-5+3*2", 11),
            ("100/10+5*2", 20), ("0+0", 0), ("1*1", 1), ("10/10", 1),
            ("5-5", 0), ("100+200", 300), ("50*4", 200), ("1000/10", 100),
        ]

    def _get_task_description(self) -> str:
        return (
            "Implement a SimpleInterpreter class with methods:\n"
            "- tokenize(expr): tokenize arithmetic expression\n"
            "- evaluate(expr): evaluate expression and return result\n"
            "Support +, -, *, /, parentheses. Operator precedence applies.\n"
            "Examples: '2+3*4'=14, '(2+3)*4'=20, '1+1'=2"
        )

    def _evaluate(self, code: str) -> Dict[str, Any]:
        passed = 0
        total = len(self.test_cases)
        errors = []
        has_syntax_error = False

        try:
            ast.parse(code)
            namespace = {}
            exec(code, namespace)

            if "SimpleInterpreter" not in namespace:
                errors.append("Class 'SimpleInterpreter' not found")
                return {
                    "loss": 1.0,
                    "passed_tests": 0,
                    "total_tests": total,
                    "errors": errors,
                    "has_syntax_error": True,
                }

            interpreter = namespace["SimpleInterpreter"]()
            for expr, expected in self.test_cases:
                try:
                    result = interpreter.evaluate(expr)
                    if result == expected:
                        passed += 1
                    else:
                        errors.append(f"Test failed: evaluate('{expr}') = {result}, expected {expected}")
                except Exception as e:
                    errors.append(f"Runtime error on '{expr}': {str(e)}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {str(e)}")
            has_syntax_error = True

        loss = 1.0 - (passed / total) if total > 0 else 1.0
        return {
            "loss": loss,
            "passed_tests": passed,
            "total_tests": total,
            "errors": errors,
            "has_syntax_error": has_syntax_error,
        }
