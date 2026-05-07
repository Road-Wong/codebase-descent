from enum import Enum
from typing import List, Dict, Any, Optional
from collections import deque


class ExperimentStatus(Enum):
    """Status labels for experiment dynamics."""
    WARMUP = "Warming Up"
    CONVERGING = "Normal Descent"
    OSCILLATING = "Limit Cycle Detected"
    STAGNATING = "Vanishing Gradient"
    DIVERGING = "Exploding Gradient"
    SOLVED = "Converged to Solution"


class DynamicsAuditor:
    """
Real-time Dynamics Auditor for Code Optimization.

This module tracks the optimization trajectory and classifies the dynamics
into different states: converging, oscillating, stagnating, or diverging.

NEW in v2.0: Added oscillation index calculation using edit distance.
"""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize auditor.
        
        Args:
            window_size: Number of recent steps to consider for analysis
        """
        self.window_size = window_size
        self.loss_history = deque(maxlen=100)
        self.code_hashes = deque(maxlen=100)
        self.code_history = []  # Store full code strings for oscillation index
        self.status_history = []
        
    def step(self, loss: float, code: str) -> ExperimentStatus:
        """
        Audit one step and return current status.
        
        Args:
            loss: Current loss value
            code: Current code state
            
        Returns:
            ExperimentStatus enum value
        """
        self.loss_history.append(loss)
        code_hash = hash(code)
        self.code_hashes.append(code_hash)
        self.code_history.append(code)  # Store full code for oscillation calculation
        
        status = self._audit()
        self.status_history.append(status)
        
        return status
    
    def _audit(self) -> ExperimentStatus:
        """
        Analyze recent history to determine current status.
        
        Returns:
            Current experiment status
        """
        # Need at least 3 steps for meaningful analysis
        if len(self.loss_history) < 3:
            return ExperimentStatus.WARMUP
        
        recent_losses = list(self.loss_history)[-self.window_size:]
        recent_hashes = list(self.code_hashes)[-self.window_size:]
        
        # Check if solved (loss = 0)
        if recent_losses[-1] == 0.0:
            return ExperimentStatus.SOLVED
        
        # 1. Detect oscillation: Code hash repeats and loss not decreasing
        if self._detect_oscillation(recent_hashes, recent_losses):
            return ExperimentStatus.OSCILLATING
        
        # 2. Detect gradient vanishing: Code unchanged but loss > 0
        if self._detect_stagnation(recent_hashes, recent_losses):
            return ExperimentStatus.STAGNATING
        
        # 3. Detect gradient explosion: Loss suddenly increases
        if self._detect_divergence(recent_losses):
            return ExperimentStatus.DIVERGING
        
        # 4. Normal convergence: Loss decreasing
        if self._detect_convergence(recent_losses):
            return ExperimentStatus.CONVERGING
        
        return ExperimentStatus.WARMUP
    
    def _detect_oscillation(self, hashes: List[int], losses: List[float]) -> bool:
        """
        Detect limit cycle: code repeats or alternates.
        
        Args:
            hashes: Recent code hashes
            losses: Recent loss values
            
        Returns:
            True if oscillating
        """
        if len(hashes) < 4:
            return False
        
        # Check for alternating pattern (2 unique hashes repeating)
        if len(set(hashes[-4:])) == 2:
            # And loss is not decreasing
            if losses[-1] >= losses[-3] * 0.95:
                return True
        
        # Check if current hash appeared recently (but not stagnant)
        current_hash = hashes[-1]
        if current_hash in hashes[-4:-1] and len(set(hashes[-3:])) > 1:
            # And loss is not improving
            if losses[-1] >= losses[-2] * 0.95:
                return True
        
        return False
    
    def _detect_stagnation(self, hashes: List[int], losses: List[float]) -> bool:
        """
        Detect vanishing gradient: code stops changing.
        
        Args:
            hashes: Recent code hashes
            losses: Recent loss values
            
        Returns:
            True if stagnating
        """
        if len(hashes) < 3:
            return False
        
        # Code hasn't changed for multiple steps
        if len(set(hashes[-3:])) == 1:  # All same hash
            # But loss is still non-zero
            if losses[-1] > 0.05:
                return True
        
        return False
    
    def _detect_divergence(self, losses: List[float]) -> bool:
        """
        Detect gradient explosion: loss increases sharply.
        
        Args:
            losses: Recent loss values
            
        Returns:
            True if diverging
        """
        if len(losses) < 2:
            return False
        
        # Loss increased by more than 50%
        if losses[-1] > losses[-2] * 1.5:
            return True
        
        # Loss increasing trend
        if len(losses) >= 3:
            if losses[-1] > losses[-2] > losses[-3]:
                return True
        
        return False
    
    def _detect_convergence(self, losses: List[float]) -> bool:
        """
        Detect normal convergence: loss decreasing.
        
        Args:
            losses: Recent loss values
            
        Returns:
            True if converging normally
        """
        if len(losses) < 2:
            return False
        
        # Loss is decreasing
        if losses[-1] < losses[-2]:
            return True
        
        # Overall trend is downward
        if len(losses) >= 4:
            avg_recent = sum(losses[-2:]) / 2
            avg_older = sum(losses[-4:-2]) / 2
            if avg_recent < avg_older * 0.9:
                return True
        
        return False
    
    def calculate_oscillation_index(self) -> float:
        """
        Calculate oscillation index based on code edit distance.
        
        Oscillation Index = (Total Edit Distance / Net Edit Distance) - 1
        
        Interpretation:
        - 0.0: Perfect straight-line optimization (no oscillation)
        - 1.0: You traveled twice the direct distance (moderate oscillation)
        - 10.0+: Severe oscillation (going in circles)
        
        If net distance is 0 (returned to start), we cap at 10.0 to avoid infinity.
        
        Returns:
            Oscillation index value
        """
        if len(self.code_history) < 2:
            return 0.0
        
        try:
            import Levenshtein
        except ImportError:
            # Fallback: use simple character-level distance
            def simple_distance(s1, s2):
                return sum(c1 != c2 for c1, c2 in zip(s1, s2)) + abs(len(s1) - len(s2))
            
            class FallbackLevenshtein:
                @staticmethod
                def distance(s1, s2):
                    return sum(c1 != c2 for c1, c2 in zip(s1, s2)) + abs(len(s1) - len(s2))
            
            Levenshtein = FallbackLevenshtein
        
        # Calculate total edit distance (sum of all consecutive changes)
        total_dist = 0
        for i in range(len(self.code_history) - 1):
            dist = Levenshtein.distance(self.code_history[i], self.code_history[i+1])
            total_dist += dist
        
        # Calculate net edit distance (start to end)
        net_dist = Levenshtein.distance(self.code_history[0], self.code_history[-1])
        
        # Handle edge case: returned to starting point
        if net_dist == 0:
            return 10.0  # Cap at 10.0 to indicate severe oscillation
        
        # Calculate oscillation index
        oscillation_index = (total_dist / net_dist) - 1.0
        
        return max(0.0, oscillation_index)  # Ensure non-negative
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the trajectory.
        
        Returns:
            Dictionary with statistics
        """
        if not self.loss_history:
            return {}
        
        losses = list(self.loss_history)
        
        # Count status occurrences
        status_counts = {}
        for status in self.status_history:
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        # Calculate metrics
        oscillation_index = self.calculate_oscillation_index()
        
        return {
            'total_steps': len(losses),
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'max_loss': max(losses),
            'oscillation_index': oscillation_index,
            'status_counts': status_counts,
            'converged': losses[-1] == 0.0
        }
    
    def reset(self):
        """Reset auditor state."""
        self.loss_history.clear()
        self.code_hashes.clear()
        self.code_history.clear()
        self.status_history.clear()
