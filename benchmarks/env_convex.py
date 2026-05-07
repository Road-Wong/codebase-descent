import ast
import re
from typing import Dict, Any, Tuple
from .abstract_env import AbstractEnvironment


class ConvexEnvironment(AbstractEnvironment):
    """
    Simple convex function environment (Base Case).
    Task: Implement a function that matches a simple mathematical formula.
    """
    
    def __init__(self):
        ground_truth = """def compute(x):
    return 2 * x + 3"""
        super().__init__(ground_truth)
        
        self.test_cases = [
            (0, 3),
            (1, 5),
            (2, 7),
            (-1, 1),
            (10, 23),
            (5, 13),
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
        Loss function: L = L_test + lambda * D_sem
        For convex case, we use simple test failure count.
        """
        eval_result = self.evaluate(code)
        failed_tests = eval_result['total_tests'] - eval_result['passed_tests']
        
        # Normalize to [0, 1]
        loss_test = failed_tests / eval_result['total_tests']
        
        # Semantic distance (simplified: based on code similarity)
        loss_sem = self._semantic_distance(code, self.c_star)
        
        # Combined loss
        lambda_weight = 0.3
        return loss_test + lambda_weight * loss_sem
    
    def evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate code against test suite."""
        passed = 0
        total = len(self.test_cases)
        errors = []
        results = []
        
        try:
            # Parse and validate syntax
            ast.parse(code)
            
            # Execute code in isolated namespace
            namespace = {}
            exec(code, namespace)
            
            if 'compute' not in namespace:
                errors.append("Function 'compute' not found")
                return {
                    'passed_tests': 0,
                    'total_tests': total,
                    'errors': errors,
                    'execution_results': []
                }
            
            compute_func = namespace['compute']
            
            # Run test cases
            for x, expected in self.test_cases:
                try:
                    result = compute_func(x)
                    if result == expected:
                        passed += 1
                        results.append({'input': x, 'expected': expected, 'actual': result, 'passed': True})
                    else:
                        results.append({'input': x, 'expected': expected, 'actual': result, 'passed': False})
                        errors.append(f"Test failed: compute({x}) = {result}, expected {expected}")
                except Exception as e:
                    results.append({'input': x, 'expected': expected, 'error': str(e), 'passed': False})
                    errors.append(f"Runtime error on input {x}: {str(e)}")
                    
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
        """
        Simplified semantic distance based on string similarity.
        In full implementation, would use CodeBERT embeddings.
        """
        # Normalize whitespace
        c1 = re.sub(r'\s+', ' ', code1.strip())
        c2 = re.sub(r'\s+', ' ', code2.strip())
        
        # Levenshtein-like distance (simplified)
        if c1 == c2:
            return 0.0
        
        # Character-level similarity
        max_len = max(len(c1), len(c2))
        if max_len == 0:
            return 0.0
            
        common = sum(1 for a, b in zip(c1, c2) if a == b)
        return 1.0 - (common / max_len)
