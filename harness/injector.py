import random
import re
import ast
from typing import Tuple, Optional


class NoiseInjector:
    """
    Controlled code perturbation for difficulty calibration.
    Places initial state c_0 at the edge of basin of attraction.
    """
    
    def __init__(self, ground_truth_code: str):
        """
        Initialize injector with ground truth code.
        
        Args:
            ground_truth_code: The optimal solution c*
        """
        self.c_star = ground_truth_code
        self.lines = ground_truth_code.split('\n')
        
    def perturb(self, difficulty_level: int, seed: Optional[int] = None) -> Tuple[str, float]:
        """
        Apply controlled perturbations based on difficulty level.
        
        Level 1 (Perturbation): Variable renaming, simple syntax breaks
                                (simulates near local minimum)
        Level 2 (Mutation): Logic operator inversion, off-by-one errors
                           (simulates saddle point)
        Level 3 (Destruction): Delete key functions, parameter shuffling
                              (simulates random initialization)
        Level 4 (Chaos): Mixed syntax+logic+structure errors
                        (simulates far from optimum)
        
        Args:
            difficulty_level: 1, 2, 3, or 4
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (perturbed_code, initial_distance)
        """
        if seed is not None:
            random.seed(seed)
        
        code = self.c_star
        
        if difficulty_level >= 1:
            code = self._inject_syntax_noise(code)
        
        if difficulty_level >= 2:
            code = self._inject_logic_inversion(code)
        
        if difficulty_level >= 3:
            code = self._inject_coupling_break(code)
        
        if difficulty_level >= 4:
            code = self._inject_chaos(code)
        
        distance = self._calculate_initial_distance(code)
        return code, distance
    
    def _inject_syntax_noise(self, code: str) -> str:
        """
        Level 1: Inject simple syntax errors and variable renaming.
        增强版: 添加缩进错误和变量重命名
        """
        # Variable renaming (more general patterns)
        code = re.sub(r'\bkey\b', 'k', code)
        code = re.sub(r'\bvalue\b', 'val', code)
        code = re.sub(r'\bcapacity\b', 'cap', code)
        code = re.sub(r'\bnode\b', 'nd', code)
        
        # Add indentation errors (ensure at least one change)
        lines = code.split('\n')
        modified_lines = []
        changed = False
        for i, line in enumerate(lines):
            if line.strip() and not changed and i > 0:
                # Guarantee at least one indentation error
                if line.startswith('    '):
                    line = '  ' + line  # Wrong indentation
                    changed = True
                elif line.startswith('        '):
                    line = '      ' + line.lstrip()  # Wrong indentation
                    changed = True
            elif line.strip() and random.random() < 0.2:
                # Additional random errors
                if line.startswith('    '):
                    line = '  ' + line
                elif line.startswith('        '):
                    line = '      ' + line.lstrip()
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)
    
    def _inject_logic_inversion(self, code: str) -> str:
        """
        Level 2: Invert logical operators and introduce off-by-one errors.
        增强版: 同时破坏多个运算符
        """
        # Change multiple operators
        code = code.replace('+', '-', 2)  # Change 2 occurrences
        code = code.replace('*', '//', 1)
        
        # Introduce off-by-one in comparisons and ranges
        code = re.sub(r'< len\(', r'<= len(', code)
        code = re.sub(r'\+ 1\b', '', code)  # Remove +1
        
        # Swap operand order in some cases
        code = re.sub(r'left = left \+ right', 'left = right + left', code)
        
        return code
    
    def _inject_coupling_break(self, code: str) -> str:
        """
        Level 3: Break coupling by modifying function signatures or deleting key parts.
        增强版: 删除关键逻辑块,打乱函数调用顺序
        """
        lines = code.split('\n')
        new_lines = []
        deleted_count = 0
        
        for i, line in enumerate(lines):
            # Delete key computation lines (not just return)
            if ('left' in line or 'right' in line) and '=' in line and deleted_count < 2:
                deleted_count += 1
                continue
            # Delete recursive calls
            if 'self._eval' in line and deleted_count < 3:
                deleted_count += 1
                continue
            new_lines.append(line)
        
        # Also mess up function signatures
        result = '\n'.join(new_lines)
        result = re.sub(r'def (\w+)\(self, tokens, pos\)', r'def \1(self, pos, tokens)', result)
        
        return result
    
    def _inject_chaos(self, code: str) -> str:
        """
        Level 4: Chaos - mix syntax, logic, and structure errors.
        最高难度: 混合所有类型的错误
        """
        lines = code.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            # Randomly skip lines
            if random.random() < 0.2 and line.strip():
                continue
            
            # Randomly duplicate lines
            if random.random() < 0.1:
                new_lines.append(line)
            
            # Add random syntax errors
            if 'def ' in line:
                line = line.replace('def ', 'def  ')  # Double space
            
            new_lines.append(line)
        
        result = '\n'.join(new_lines)
        
        # Mess up all operators
        result = result.replace('+', '++', 1)
        result = result.replace('*', '**', 1)
        result = result.replace('==', '=', 1)
        
        return result
    
    def _calculate_initial_distance(self, code: str) -> float:
        """
        Calculate normalized edit distance from ground truth.
        
        Args:
            code: Perturbed code
            
        Returns:
            Distance in [0, 1]
        """
        c1 = re.sub(r'\s+', '', code)
        c2 = re.sub(r'\s+', '', self.c_star)
        
        # Simple character-level distance
        max_len = max(len(c1), len(c2))
        if max_len == 0:
            return 0.0
        
        # Count matching characters
        matches = sum(1 for a, b in zip(c1, c2) if a == b)
        distance = 1.0 - (matches / max_len)
        
        return distance
    
    def calibrate(self, env, agent, trials: int = 5) -> int:
        """
        Automatic calibration: Find difficulty level where T=0 fails but T=0.7 succeeds.
        
        Args:
            env: Environment to test on
            agent: Agent to use for testing
            trials: Number of trials per level
            
        Returns:
            Calibrated difficulty level (1, 2, or 3)
        """
        for level in [1, 2, 3]:
            greedy_success = 0
            annealed_success = 0
            
            for _ in range(trials):
                # Test with greedy (T=0)
                perturbed_code, _ = self.perturb(level)
                agent.temperature = 0.0
                result = self._quick_test(env, agent, perturbed_code, max_steps=5)
                if result:
                    greedy_success += 1
                
                # Test with annealing (T=0.7)
                perturbed_code, _ = self.perturb(level)
                agent.temperature = 0.7
                result = self._quick_test(env, agent, perturbed_code, max_steps=10)
                if result:
                    annealed_success += 1
            
            greedy_rate = greedy_success / trials
            annealed_rate = annealed_success / trials
            
            # Target: greedy fails (<20%), annealed succeeds (>80%)
            if greedy_rate < 0.2 and annealed_rate > 0.8:
                return level
        
        return -1  # Calibration failed
    
    def _quick_test(self, env, agent, initial_code: str, max_steps: int) -> bool:
        """
        Quick test to check if agent can solve from initial code.
        
        Returns:
            True if solved within max_steps
        """
        code = initial_code
        task_desc = "Fix the code to pass all tests"
        
        for _ in range(max_steps):
            loss, info = env.step(code)
            
            if env.is_solved(code):
                return True
            
            # Generate next code
            code = agent.step(task_desc, code, info)
        
        return False
