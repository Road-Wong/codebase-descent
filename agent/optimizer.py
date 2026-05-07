from typing import List, Dict, Any, Optional
from .llm_kernel import LLMKernel
from .memory_buffer import MemoryBuffer


class CodeOptimizer:
    """
    Implements optimization strategies for code generation.
    Analogous to SGD/Momentum/Adam optimizers in neural networks.
    """
    
    def __init__(
        self,
        llm_kernel: LLMKernel,
        strategy: str = "sgd",
        context_window: int = 5,
        temperature: float = 0.7
    ):
        """
        Initialize optimizer.
        
        Args:
            llm_kernel: LLM kernel for code generation
            strategy: Optimization strategy ('sgd', 'momentum', 'adaptive')
            context_window: Number of previous iterations to keep (batch size analog)
            temperature: Sampling temperature (learning rate analog)
        """
        self.llm = llm_kernel
        self.strategy = strategy
        self.context_window = context_window
        self.temperature = temperature
        self.memory = MemoryBuffer(max_size=context_window)
        
    def step(
        self,
        task_description: str,
        current_code: str,
        feedback: Dict[str, Any],
        use_thinking: bool = False
    ) -> str:
        """
        Perform one optimization step.
        
        Args:
            task_description: Task specification
            current_code: Current code state c_t
            feedback: Test results and error messages
            use_thinking: Whether to use chain-of-thought
            
        Returns:
            Updated code c_{t+1}
        """
        # Format error feedback
        error_msg = self._format_feedback(feedback)
        
        # Get history for momentum
        history = self.memory.get_history()
        
        # Generate next code
        if use_thinking:
            result = self.llm.generate_with_thinking(
                task_description=task_description,
                current_code=current_code,
                error_feedback=error_msg,
                temperature=self.temperature
            )
            next_code = result['code']
            thinking = result['thinking']
        else:
            next_code = self.llm.generate_code(
                task_description=task_description,
                current_code=current_code,
                error_feedback=error_msg,
                history=history,
                temperature=self.temperature
            )
            thinking = None
        
        # Update memory
        self.memory.add({
            'code': current_code,
            'feedback': error_msg,
            'thinking': thinking
        })
        
        return next_code
    
    def _format_feedback(self, feedback: Dict[str, Any]) -> str:
        """Format feedback into error message string."""
        if not feedback.get('errors'):
            return "All tests passed!"
        
        msg = f"Passed {feedback['passed_tests']}/{feedback['total_tests']} tests.\n\n"
        msg += "Errors:\n"
        for error in feedback['errors'][:5]:  # Show first 5 errors
            msg += f"- {error}\n"
        
        return msg
    
    def anneal_temperature(self, step: int, max_steps: int, gamma: float = 1.0):
        """
        Implement temperature annealing (learning rate decay).
        
        Args:
            step: Current step number
            max_steps: Maximum number of steps
            gamma: Decay exponent
        """
        initial_temp = self.temperature
        self.temperature = initial_temp * (1 - step / max_steps) ** gamma
        
    def reset(self):
        """Reset optimizer state."""
        self.memory.clear()


class AdaptiveOptimizer(CodeOptimizer):
    """
    Adaptive optimizer that adjusts temperature based on progress.
    Similar to Adam optimizer with adaptive learning rates.
    """
    
    def __init__(self, llm_kernel: LLMKernel, **kwargs):
        super().__init__(llm_kernel, strategy="adaptive", **kwargs)
        self.loss_history = []
        self.temp_min = 0.2
        self.temp_max = 1.2
        
    def step(self, task_description: str, current_code: str, feedback: Dict[str, Any], **kwargs) -> str:
        """Step with adaptive temperature adjustment."""
        # Track loss
        loss = 1.0 - (feedback['passed_tests'] / feedback['total_tests'])
        self.loss_history.append(loss)
        
        # Adjust temperature based on progress
        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            
            # If stuck (no improvement), increase temperature (exploration)
            if recent_losses[-1] >= recent_losses[-2] >= recent_losses[-3]:
                self.temperature = min(self.temp_max, self.temperature * 1.2)
            # If improving, decrease temperature (exploitation)
            elif recent_losses[-1] < recent_losses[-2]:
                self.temperature = max(self.temp_min, self.temperature * 0.9)
        
        return super().step(task_description, current_code, feedback, **kwargs)
