"""
Graph Algorithm Environment - Complex task requiring deep reasoning.

This environment tests the agent's ability to implement graph traversal algorithms
with proper state management, recursion handling, and edge case consideration.

Task Complexity:
- Requires understanding of graph data structures
- Multiple interdependent functions
- Recursion or stack management
- Visited state tracking is critical
- Many edge cases (empty graph, cycles, disconnected components)
"""

from typing import Dict, Any, Tuple, Optional, List
from .abstract_env import AbstractEnvironment


class GraphEnvironment(AbstractEnvironment):
    """
    Environment for implementing graph traversal algorithms.
    
    The task is to implement a Graph class with:
    - __init__(): Initialize empty graph
    - add_edge(u, v): Add directed edge from u to v
    - dfs(start): Return DFS traversal order starting from node
    - bfs(start): Return BFS traversal order starting from node
    - has_path(start, end): Check if path exists between nodes
    
    This is a harder task because:
    1. Requires understanding of graph algorithms
    2. Proper state management (visited nodes)
    3. Recursion or queue management
    4. Multiple edge cases
    5. Different algorithms require different approaches
    """
    
    def __init__(self, ground_truth: Optional[str] = None):
        """Initialize Graph environment."""
        self.c_star = ground_truth or self._get_ground_truth()
        super().__init__(self.c_star)
        
        # Test cases: (graph_edges, operations, expected_results)
        self.test_cases = [
            # Simple linear graph: 0 -> 1 -> 2
            {
                "edges": [(0, 1), (1, 2)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 2)),
                    ("has_path", (2, 0))
                ],
                "expected": [
                    [0, 1, 2],
                    [0, 1, 2],
                    True,
                    False
                ]
            },
            
            # Simple tree: 0 -> 1, 0 -> 2
            {
                "edges": [(0, 1), (0, 2)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 1)),
                    ("has_path", (1, 2))
                ],
                "expected": [
                    [0, 1, 2],  # DFS can be [0,1,2] or [0,2,1]
                    [0, 1, 2],  # BFS order
                    True,
                    False
                ]
            },
            
            # Graph with cycle: 0 -> 1 -> 2 -> 0
            {
                "edges": [(0, 1), (1, 2), (2, 0)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 2)),
                    ("has_path", (2, 0))
                ],
                "expected": [
                    [0, 1, 2],
                    [0, 1, 2],
                    True,
                    True
                ]
            },
            
            # Disconnected graph
            {
                "edges": [(0, 1), (2, 3)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (2,)),
                    ("has_path", (0, 1)),
                    ("has_path", (0, 3))
                ],
                "expected": [
                    [0, 1],
                    [2, 3],
                    True,
                    False
                ]
            },
            
            # Single node
            {
                "edges": [],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 0))
                ],
                "expected": [
                    [0],
                    [0],
                    True
                ]
            },
            
            # Complex graph: diamond shape
            # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
            {
                "edges": [(0, 1), (0, 2), (1, 3), (2, 3)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 3)),
                    ("has_path", (3, 0))
                ],
                "expected": [
                    [0, 1, 3, 2],  # One valid DFS order
                    [0, 1, 2, 3],  # BFS order
                    True,
                    False
                ]
            },
            
            # Larger graph with multiple paths
            {
                "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 4)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 4)),
                    ("has_path", (4, 0))
                ],
                "expected": [
                    [0, 1, 3, 4, 2],  # One valid DFS order
                    [0, 1, 2, 3, 4],  # BFS order
                    True,
                    False
                ]
            },
            
            # Self-loop
            {
                "edges": [(0, 0), (0, 1)],
                "operations": [
                    ("dfs", (0,)),
                    ("bfs", (0,)),
                    ("has_path", (0, 1))
                ],
                "expected": [
                    [0, 1],
                    [0, 1],
                    True
                ]
            }
        ]
    
    def _get_ground_truth(self) -> str:
        """Return the ground truth implementation."""
        return """class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    
    def dfs(self, start):
        visited = set()
        result = []
        
        def dfs_helper(node):
            if node in visited:
                return
            visited.add(node)
            result.append(node)
            if node in self.graph:
                for neighbor in self.graph[node]:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def bfs(self, start):
        visited = set()
        queue = [start]
        result = []
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            result.append(node)
            if node in self.graph:
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def has_path(self, start, end):
        if start == end:
            return True
        visited = set()
        
        def dfs_path(node):
            if node == end:
                return True
            if node in visited:
                return False
            visited.add(node)
            if node in self.graph:
                for neighbor in self.graph[node]:
                    if dfs_path(neighbor):
                        return True
            return False
        
        return dfs_path(start)
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
        
        # Penalty for syntax errors
        syntax_penalty = 0.2 if info['has_syntax_error'] else 0.0
        
        # Semantic distance
        semantic_distance = self._semantic_distance(code)
        
        # Combined loss
        loss = test_loss + syntax_penalty + 0.1 * semantic_distance
        
        return loss
    
    def evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate code against test cases."""
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        has_syntax_error = False
        
        try:
            namespace = {}
            exec(code, namespace)
            
            if 'Graph' not in namespace:
                errors.append("Graph class not found")
                has_syntax_error = True
                return {
                    'passed_tests': 0,
                    'total_tests': total_tests,
                    'errors': errors,
                    'has_syntax_error': has_syntax_error
                }
            
            Graph = namespace['Graph']
            
            # Run each test case
            for i, test_case in enumerate(self.test_cases):
                try:
                    g = Graph()
                    for u, v in test_case['edges']:
                        g.add_edge(u, v)
                    
                    results = []
                    for op, args in test_case['operations']:
                        if op == 'dfs':
                            result = g.dfs(args[0])
                        elif op == 'bfs':
                            result = g.bfs(args[0])
                        elif op == 'has_path':
                            result = g.has_path(args[0], args[1])
                        results.append(result)
                    
                    # Check results (allowing for valid DFS variations)
                    if self._results_match(results, test_case['expected'], test_case['operations']):
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
            'errors': errors[:3],
            'has_syntax_error': has_syntax_error
        }
    
    def _results_match(self, results: List, expected: List, operations: List) -> bool:
        """Check if results match expected, allowing for valid DFS variations."""
        if len(results) != len(expected):
            return False
        
        for i, (result, exp, (op, _)) in enumerate(zip(results, expected, operations)):
            if op == 'dfs':
                # DFS can have multiple valid orders, check if it's a valid traversal
                if not self._is_valid_dfs(result, exp):
                    return False
            else:
                # For BFS and has_path, exact match required
                if result != exp:
                    return False
        
        return True
    
    def _is_valid_dfs(self, result: List, expected: List) -> bool:
        """Check if DFS result is valid (same nodes, valid order)."""
        # Must visit same nodes
        if set(result) != set(expected):
            return False
        # Must have same length
        if len(result) != len(expected):
            return False
        return True
    
    def _semantic_distance(self, code: str) -> float:
        """Calculate semantic distance from ground truth."""
        has_init = '__init__' in code
        has_add_edge = 'add_edge' in code
        has_dfs = 'def dfs' in code
        has_bfs = 'def bfs' in code
        has_has_path = 'has_path' in code
        has_visited = 'visited' in code
        has_queue = 'queue' in code or 'deque' in code
        
        components = [has_init, has_add_edge, has_dfs, has_bfs, has_has_path, has_visited, has_queue]
        present_count = sum(components)
        
        distance = 1.0 - (present_count / len(components))
        return distance
    
    def reset(self):
        """Reset environment state."""
        self.step_count = 0
    
    def is_solved(self, code: str) -> bool:
        """Check if code solves the task."""
        info = self.evaluate(code)
        return info['passed_tests'] == info['total_tests']
