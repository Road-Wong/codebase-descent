import ast
import re
from typing import Dict, Any, Tuple, List
from .abstract_env import AbstractEnvironment


class SaddleEnvironment(AbstractEnvironment):
    """
    Complex coupled system environment (Main Case).
    Task: Implement a simple interpreter with coupled Lexer and Parser.
    Modifying one component affects the other (high coupling).
    """
    
    def __init__(self):
        ground_truth = """class SimpleInterpreter:
    def __init__(self):
        self.tokens = []
        
    def tokenize(self, expr):
        # Tokenize expression
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
        super().__init__(ground_truth)
        
        # Test cases: (expression, expected_result)
        # 扩展至20+测试用例,覆盖更多边界情况
        self.test_cases = [
            # 基础运算
            ("1+1", 2),
            ("2*3", 6),
            ("10-5", 5),
            ("8/2", 4),
            
            # 运算符优先级
            ("2+3*4", 14),
            ("(2+3)*4", 20),
            ("2*3+4*5", 26),
            ("5+3*2-4", 7),
            
            # 多重运算
            ("10-2-3", 5),
            ("1+2+3+4", 10),
            ("100/10/2", 5),
            ("(1+2)*(3+4)", 21),
            
            # 嵌套括号
            ("((1+2)*3)+4", 13),
            ("(1+(2*3))+4", 11),
            ("((2+3)*(4+5))", 45),
            
            # 复杂表达式
            ("1+2*3+4*5", 27),
            ("(1+2)*(3+4)*(5+6)", 231),
            ("10-5+3*2", 11),
            ("100/10+5*2", 20),
            
            # 边界情况
            ("0+0", 0),
            ("1*1", 1),
            ("10/10", 1),
            ("5-5", 0),
            
            # 大数计算
            ("100+200", 300),
            ("50*4", 200),
            ("1000/10", 100),
        ]
        
    def step(self, code: str) -> Tuple[float, Dict[str, Any]]:
        """Execute one step and return loss and info."""
        self.step_count += 1
        loss = self.get_loss(code)
        info = self.evaluate(code)
        info['step'] = self.step_count
        info['loss'] = loss
        return loss, info
    
    def get_loss(self, code: str) -> float:
        """
        Loss function with coupling penalty.
        L = L_test + lambda * ||c - c_prev||
        """
        eval_result = self.evaluate(code)
        failed_tests = eval_result['total_tests'] - eval_result['passed_tests']
        
        # Test loss (normalized)
        loss_test = failed_tests / eval_result['total_tests']
        
        # Semantic distance
        loss_sem = self._semantic_distance(code, self.c_star)
        
        # Stability constraint (penalize large changes)
        lambda_weight = 0.2
        return loss_test + lambda_weight * loss_sem
    
    def evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate interpreter implementation."""
        passed = 0
        total = len(self.test_cases)
        errors = []
        results = []
        
        try:
            # Parse and validate syntax
            ast.parse(code)
            
            # Execute code
            namespace = {}
            exec(code, namespace)
            
            if 'SimpleInterpreter' not in namespace:
                errors.append("Class 'SimpleInterpreter' not found")
                return {
                    'passed_tests': 0,
                    'total_tests': total,
                    'errors': errors,
                    'execution_results': []
                }
            
            # Create interpreter instance
            interpreter = namespace['SimpleInterpreter']()
            
            # Run test cases
            for expr, expected in self.test_cases:
                try:
                    result = interpreter.evaluate(expr)
                    if result == expected:
                        passed += 1
                        results.append({'expr': expr, 'expected': expected, 'actual': result, 'passed': True})
                    else:
                        results.append({'expr': expr, 'expected': expected, 'actual': result, 'passed': False})
                        errors.append(f"Test failed: evaluate('{expr}') = {result}, expected {expected}")
                except Exception as e:
                    results.append({'expr': expr, 'expected': expected, 'error': str(e), 'passed': False})
                    errors.append(f"Runtime error on '{expr}': {str(e)}")
                    
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
        except Exception as e:
            errors.append(f"Execution error: {str(e)}")
            
        return {
            'passed_tests': passed,
            'total_tests': total,
            'errors': errors,
            'execution_results': results
        }
    
    def _semantic_distance(self, code1: str, code2: str) -> float:
        """Calculate semantic distance between code snippets."""
        c1 = re.sub(r'\s+', ' ', code1.strip())
        c2 = re.sub(r'\s+', ' ', code2.strip())
        
        if c1 == c2:
            return 0.0
        
        max_len = max(len(c1), len(c2))
        if max_len == 0:
            return 0.0
            
        common = sum(1 for a, b in zip(c1, c2) if a == b)
        return 1.0 - (common / max_len)
