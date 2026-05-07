"""
LRU Cache Environment - A stateful task requiring multi-step reasoning.

This environment tests the agent's ability to implement a Least Recently Used (LRU) cache
with proper state management, edge case handling, and efficient operations.

Task Complexity:
- Requires maintaining both a doubly-linked list and a hash map
- Multiple interdependent methods
- Non-obvious edge cases (capacity=1, get/put ordering)
- State consistency is critical
"""

from typing import Dict, Any, Tuple, Optional
from .abstract_env import AbstractEnvironment


class LRUCacheEnvironment(AbstractEnvironment):
    """
    Environment for implementing an LRU Cache.
    
    The task is to implement a class with:
    - __init__(capacity): Initialize with given capacity
    - get(key): Return value if exists, else -1 (marks as recently used)
    - put(key, value): Insert or update, evict LRU if at capacity
    
    This is a harder task because:
    1. Requires understanding of data structure design
    2. Multiple methods must work together correctly
    3. Edge cases are non-trivial
    4. State management is critical
    """
    
    def __init__(self, ground_truth: Optional[str] = None):
        """Initialize LRU Cache environment."""
        # Ground truth implementation
        self.c_star = ground_truth or self._get_ground_truth()
        super().__init__(self.c_star)
        
        # Test cases: (operations, expected_results)
        # Format: [("operation", args), expected_result]
        self.test_cases = [
            # Basic operations
            {
                "capacity": 2,
                "operations": [
                    ("put", (1, 1)),
                    ("put", (2, 2)),
                    ("get", (1,)),
                    ("put", (3, 3)),
                    ("get", (2,)),
                    ("put", (4, 4)),
                    ("get", (1,)),
                    ("get", (3,)),
                    ("get", (4,))
                ],
                "expected": [None, None, 1, None, -1, None, -1, 3, 4]
            },
            
            # Capacity 1 (edge case)
            {
                "capacity": 1,
                "operations": [
                    ("put", (2, 1)),
                    ("get", (2,)),
                    ("put", (3, 2)),
                    ("get", (2,)),
                    ("get", (3,))
                ],
                "expected": [None, 1, None, -1, 2]
            },
            
            # Update existing key
            {
                "capacity": 2,
                "operations": [
                    ("put", (1, 1)),
                    ("put", (2, 2)),
                    ("get", (1,)),
                    ("put", (1, 10)),
                    ("get", (1,)),
                    ("put", (3, 3)),
                    ("get", (2,))
                ],
                "expected": [None, None, 1, None, 10, None, -1]
            },
            
            # Get updates recency
            {
                "capacity": 2,
                "operations": [
                    ("put", (1, 1)),
                    ("put", (2, 2)),
                    ("get", (1,)),
                    ("put", (3, 3)),
                    ("get", (1,)),
                    ("get", (2,))
                ],
                "expected": [None, None, 1, None, 1, -1]
            },
            
            # Larger capacity
            {
                "capacity": 3,
                "operations": [
                    ("put", (1, 1)),
                    ("put", (2, 2)),
                    ("put", (3, 3)),
                    ("put", (4, 4)),
                    ("get", (1,)),
                    ("get", (2,)),
                    ("get", (3,)),
                    ("get", (4,))
                ],
                "expected": [None, None, None, None, -1, 2, 3, 4]
            },
            
            # Multiple updates
            {
                "capacity": 2,
                "operations": [
                    ("put", (1, 1)),
                    ("put", (1, 2)),
                    ("put", (1, 3)),
                    ("get", (1,)),
                    ("put", (2, 2)),
                    ("get", (1,))
                ],
                "expected": [None, None, None, 3, None, 3]
            },
            
            # Empty gets
            {
                "capacity": 2,
                "operations": [
                    ("get", (1,)),
                    ("put", (1, 1)),
                    ("get", (1,)),
                    ("get", (2,))
                ],
                "expected": [-1, None, 1, -1]
            },
            
            # Complex eviction pattern
            {
                "capacity": 3,
                "operations": [
                    ("put", (1, 1)),
                    ("put", (2, 2)),
                    ("put", (3, 3)),
                    ("get", (1,)),
                    ("put", (4, 4)),
                    ("get", (2,)),
                    ("put", (5, 5)),
                    ("get", (1,)),
                    ("get", (3,)),
                    ("get", (4,))
                ],
                "expected": [None, None, None, 1, None, -1, None, 1, -1, 4]
            }
        ]
    
    def _get_ground_truth(self) -> str:
        """Return the ground truth implementation."""
        return """class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key not in self.cache:
            return -1
        # Move to end (most recently used)
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Evict least recently used
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.order.append(key)
"""
    
    def step(self, code: str) -> Tuple[float, Dict[str, Any]]:
        """Execute one step and return loss and info."""
        self.step_count += 1
        loss = self.get_loss(code)
        info = self.evaluate(code)
        info['step'] = self.step_count
        info['loss'] = loss
        return loss, info
    
    def get_loss(self, code: str) -> float:
        """Calculate loss for given code."""
        info = self.evaluate(code)
        
        # Base loss from failed tests
        test_loss = 1.0 - (info['passed_tests'] / info['total_tests'])
        
        # Penalty for syntax errors (can't even run)
        syntax_penalty = 0.2 if info['has_syntax_error'] else 0.0
        
        # Semantic distance (simple metric)
        semantic_distance = self._semantic_distance(code)
        
        # Combined loss
        loss = test_loss + syntax_penalty + 0.1 * semantic_distance
        
        return loss
    
    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        Evaluate code against test cases.
        
        Returns:
            Dictionary with evaluation results
        """
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        has_syntax_error = False
        
        # Try to execute the code
        try:
            # Create a namespace for execution
            namespace = {}
            exec(code, namespace)
            
            # Check if LRUCache class exists
            if 'LRUCache' not in namespace:
                errors.append("LRUCache class not found")
                has_syntax_error = True
                return {
                    'passed_tests': 0,
                    'total_tests': total_tests,
                    'errors': errors,
                    'has_syntax_error': has_syntax_error
                }
            
            LRUCache = namespace['LRUCache']
            
            # Run each test case
            for i, test_case in enumerate(self.test_cases):
                try:
                    cache = LRUCache(test_case['capacity'])
                    results = []
                    
                    for op, args in test_case['operations']:
                        if op == 'get':
                            result = cache.get(args[0])
                            results.append(result)
                        elif op == 'put':
                            result = cache.put(args[0], args[1])
                            results.append(result)
                    
                    # Check if results match expected
                    if results == test_case['expected']:
                        passed_tests += 1
                    else:
                        errors.append(f"Test {i+1} failed: expected {test_case['expected']}, got {results}")
                        
                except Exception as e:
                    errors.append(f"Test {i+1} error: {str(e)}")
                    
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {str(e)}")
            has_syntax_error = True
        
        return {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'errors': errors[:3],  # Limit to first 3 errors
            'has_syntax_error': has_syntax_error
        }
    
    def _semantic_distance(self, code: str) -> float:
        """
        Calculate semantic distance from ground truth.
        Simple heuristic based on key components.
        """
        # Check for key components
        has_init = '__init__' in code
        has_get = 'def get' in code
        has_put = 'def put' in code
        has_cache = 'cache' in code or 'dict' in code
        has_order = 'order' in code or 'list' in code
        
        components = [has_init, has_get, has_put, has_cache, has_order]
        present_count = sum(components)
        
        # Distance based on missing components
        distance = 1.0 - (present_count / len(components))
        
        return distance
    
    def reset(self):
        """Reset environment state."""
        self.step_count = 0
    
    def is_solved(self, code: str) -> bool:
        """Check if code solves the task."""
        info = self.evaluate(code)
        return info['passed_tests'] == info['total_tests']
