"""
Optimization strategies for code generation.
Analogous to SGD/Momentum/Adam optimizers in neural networks.
"""

from typing import List, Dict, Any, Optional

from core.agent_base import AbstractAgent
from core.types import Action, ActionFormat, Observation
from core.diff_utils import make_diff_action
from .llm_kernel import LLMKernel
from .memory_buffer import MemoryBuffer


class CodeOptimizer(AbstractAgent):
    """
    Implements optimization strategies for code generation.
    Supports SGD, Momentum, and Adaptive variants.
    """

    def __init__(
        self,
        llm_kernel: LLMKernel,
        strategy: str = "sgd",
        context_window: int = 5,
        temperature: float = 0.7,
    ):
        self.llm = llm_kernel
        self.strategy = strategy
        self.context_window = context_window
        self.temperature = temperature
        self.memory = MemoryBuffer(max_size=context_window)

    # ------------------------------------------------------------------
    # New RL interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Produce a unified diff Action from the current Observation."""
        feedback = {
            "passed_tests": observation.passed_tests,
            "total_tests": observation.total_tests,
            "errors": observation.errors,
        }
        new_code = self.step(
            task_description=observation.task_description,
            current_code=observation.current_code,
            feedback=feedback,
        )
        return make_diff_action(
            old_code=observation.current_code,
            new_code=new_code,
        )

    def reset(self) -> None:
        self.memory.clear()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def step(
        self,
        task_description: str,
        current_code: str,
        feedback: Dict[str, Any],
        use_thinking: bool = False,
    ) -> str:
        error_msg = self._format_feedback(feedback)
        history = self.memory.get_history()

        if use_thinking:
            result = self.llm.generate_with_thinking(
                task_description=task_description,
                current_code=current_code,
                error_feedback=error_msg,
                temperature=self.temperature,
            )
            next_code = result["code"]
            thinking = result["thinking"]
        else:
            next_code = self.llm.generate_code(
                task_description=task_description,
                current_code=current_code,
                error_feedback=error_msg,
                history=history,
                temperature=self.temperature,
            )
            thinking = None

        self.memory.add({
            "code": current_code,
            "feedback": error_msg,
            "thinking": thinking,
        })
        return next_code

    def _format_feedback(self, feedback: Dict[str, Any]) -> str:
        if not feedback.get("errors"):
            return "All tests passed!"
        msg = f"Passed {feedback['passed_tests']}/{feedback['total_tests']} tests.\n\n"
        msg += "Errors:\n"
        for error in feedback["errors"][:5]:
            msg += f"- {error}\n"
        return msg

    def anneal_temperature(self, step: int, max_steps: int, gamma: float = 1.0):
        initial_temp = self.temperature
        self.temperature = initial_temp * (1 - step / max_steps) ** gamma


class AdaptiveOptimizer(CodeOptimizer):
    """
    Adaptive optimizer that adjusts temperature based on progress.
    Similar to Adam optimizer with adaptive learning rates.
    """

    def __init__(self, llm_kernel: LLMKernel, **kwargs):
        super().__init__(llm_kernel, strategy="adaptive", **kwargs)
        self.loss_history: List[float] = []
        self.temp_min = 0.2
        self.temp_max = 1.2

    def step(self, task_description: str, current_code: str, feedback: Dict[str, Any], **kwargs) -> str:
        loss = 1.0 - (feedback["passed_tests"] / feedback["total_tests"])
        self.loss_history.append(loss)

        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            if recent_losses[-1] >= recent_losses[-2] >= recent_losses[-3]:
                self.temperature = min(self.temp_max, self.temperature * 1.2)
            elif recent_losses[-1] < recent_losses[-2]:
                self.temperature = max(self.temp_min, self.temperature * 0.9)

        return super().step(task_description, current_code, feedback, **kwargs)
