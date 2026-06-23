"""
Graph Algorithm Environment — Complex reasoning task.

8 test cases covering: linear graph, tree, cycle, disconnected,
single node, diamond, larger graph, self-loop.
"""

from typing import Dict, Any, List

from core.env_base import AbstractEnvironment


class GraphEnvironment(AbstractEnvironment):
    """Environment for implementing graph traversal algorithms."""

    def __init__(self, obfuscation_level: int = 0, max_steps: int = 20):
        ground_truth = """class Graph:
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
        super().__init__(
            ground_truth_code=ground_truth,
            max_steps=max_steps,
            obfuscation_level=obfuscation_level,
        )
        self.test_cases = [
            {"edges": [(0,1),(1,2)], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,2)),("has_path",(2,0))], "expected": [[0,1,2],[0,1,2],True,False]},
            {"edges": [(0,1),(0,2)], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,1)),("has_path",(1,2))], "expected": [[0,1,2],[0,1,2],True,False]},
            {"edges": [(0,1),(1,2),(2,0)], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,2)),("has_path",(2,0))], "expected": [[0,1,2],[0,1,2],True,True]},
            {"edges": [(0,1),(2,3)], "operations": [("dfs",(0,)),("bfs",(2,)),("has_path",(0,1)),("has_path",(0,3))], "expected": [[0,1],[2,3],True,False]},
            {"edges": [], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,0))], "expected": [[0],[0],True]},
            {"edges": [(0,1),(0,2),(1,3),(2,3)], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,3)),("has_path",(3,0))], "expected": [[0,1,3,2],[0,1,2,3],True,False]},
            {"edges": [(0,1),(0,2),(1,3),(2,3),(3,4),(1,4)], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,4)),("has_path",(4,0))], "expected": [[0,1,3,4,2],[0,1,2,3,4],True,False]},
            {"edges": [(0,0),(0,1)], "operations": [("dfs",(0,)),("bfs",(0,)),("has_path",(0,1))], "expected": [[0,1],[0,1],True]},
        ]

    def _get_task_description(self) -> str:
        return (
            "Implement a Graph class with:\n"
            "- __init__(): Initialize empty graph\n"
            "- add_edge(u, v): Add directed edge from u to v\n"
            "- dfs(start): Return DFS traversal order\n"
            "- bfs(start): Return BFS traversal order\n"
            "- has_path(start, end): Check if path exists"
        )

    def _evaluate(self, code: str) -> Dict[str, Any]:
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        has_syntax_error = False

        try:
            namespace = {}
            exec(code, namespace)

            if "Graph" not in namespace:
                errors.append("Graph class not found")
                return {
                    "loss": 1.0, "passed_tests": 0, "total_tests": total_tests,
                    "errors": errors, "has_syntax_error": True,
                }

            Graph = namespace["Graph"]
            for i, test_case in enumerate(self.test_cases):
                try:
                    g = Graph()
                    for u, v in test_case["edges"]:
                        g.add_edge(u, v)
                    results = []
                    for op, args in test_case["operations"]:
                        if op == "dfs":
                            results.append(g.dfs(args[0]))
                        elif op == "bfs":
                            results.append(g.bfs(args[0]))
                        elif op == "has_path":
                            results.append(g.has_path(args[0], args[1]))
                    if self._results_match(results, test_case["expected"], test_case["operations"]):
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

        loss = 1.0 - (passed_tests / total_tests) if total_tests > 0 else 1.0
        return {
            "loss": loss, "passed_tests": passed_tests, "total_tests": total_tests,
            "errors": errors[:3], "has_syntax_error": has_syntax_error,
        }

    def _results_match(self, results: List, expected: List, operations: List) -> bool:
        if len(results) != len(expected):
            return False
        for result, exp, (op, _) in zip(results, expected, operations):
            if op == "dfs":
                if set(result) != set(exp) or len(result) != len(exp):
                    return False
            else:
                if result != exp:
                    return False
        return True
