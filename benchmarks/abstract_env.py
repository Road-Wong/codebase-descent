from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class AbstractEnvironment(ABC):
    """
    Abstract base class for code optimization environments.
    Defines the interface for Loss Landscape (代码流形).
    """
    
    def __init__(self, ground_truth_code: str):
        """
        Initialize the environment with ground truth code.
        
        Args:
            ground_truth_code: The optimal solution c*
        """
        self.c_star = ground_truth_code
        self.step_count = 0
        
    @abstractmethod
    def step(self, code: str) -> Tuple[float, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            code: Current code state c_t
            
        Returns:
            Tuple of (loss, info_dict)
            - loss: L(c_t) - the loss value
            - info_dict: Additional information (test results, errors, etc.)
        """
        pass
    
    @abstractmethod
    def get_loss(self, code: str) -> float:
        """
        Calculate the loss function L(c).
        
        Args:
            code: Code to evaluate
            
        Returns:
            Loss value (0 = optimal, higher = worse)
        """
        pass
    
    @abstractmethod
    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        Evaluate code against test suite.
        
        Args:
            code: Code to evaluate
            
        Returns:
            Dictionary containing:
            - passed_tests: Number of tests passed
            - total_tests: Total number of tests
            - errors: List of error messages
            - execution_results: Detailed test results
        """
        pass
    
    def reset(self):
        """Reset environment to initial state."""
        self.step_count = 0
        
    def is_solved(self, code: str) -> bool:
        """
        Check if the code solves the task.
        
        Args:
            code: Code to check
            
        Returns:
            True if all tests pass
        """
        result = self.evaluate(code)
        return result['passed_tests'] == result['total_tests']
