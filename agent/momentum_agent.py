"""
Semantic Momentum Optimizer (SMO)

This module implements the core innovation of our paper: maintaining a "semantic momentum"
in natural language space to stabilize LLM-based code optimization.

Key Idea:
- Instead of treating each iteration independently, we maintain a momentum vector
- The momentum is a natural language summary of the "optimization direction"
- This acts as a low-pass filter, suppressing high-frequency oscillations
"""

from typing import Dict, Any, Optional, List
from .llm_kernel import LLMKernel


class SemanticMomentumOptimizer:
    """
    Semantic Momentum Optimizer - The core algorithm of our paper.
    
    This optimizer maintains a momentum state in natural language that:
    1. Summarizes the overall optimization direction
    2. Filters out noisy, contradictory suggestions
    3. Prevents oscillation by enforcing consistency
    
    Mathematical Analogy:
        m_{t+1} = β * m_t + (1-β) * g_t
    where:
        m_t: momentum at step t (natural language)
        g_t: gradient at step t (extracted from current feedback)
        β: momentum coefficient (0.0 = no momentum, 0.9 = high inertia)
    """
    
    def __init__(
        self,
        llm_kernel: LLMKernel,
        beta: float = 0.7,
        temperature: float = 0.7,
        max_history: int = 3
    ):
        """
        Initialize the Semantic Momentum Optimizer.
        
        Args:
            llm_kernel: LLM interface for code generation
            beta: Momentum coefficient (0.0-1.0)
            temperature: LLM sampling temperature
            max_history: Maximum number of recent steps to keep in context
        """
        self.llm = llm_kernel
        self.beta = beta
        self.temperature = temperature
        self.max_history = max_history
        
        # Momentum state (natural language)
        self.momentum_text = "No momentum yet. Starting fresh optimization."
        
        # History for analysis
        self.gradient_history: List[str] = []
        self.code_history: List[str] = []
        self.loss_history: List[float] = []
    
    def extract_gradient(self, current_code: str, feedback: Dict[str, Any]) -> str:
        """
        Extract semantic gradient from feedback.
        
        This is Step 1 of the momentum update: convert raw feedback into
        a structured natural language description of "what needs to change".
        
        Args:
            current_code: Current code state
            feedback: Test results and error messages
            
        Returns:
            Natural language gradient description
        """
        # Format feedback for LLM
        error_summary = self._format_feedback(feedback)
        
        prompt = f"""You are analyzing code optimization feedback to extract the "semantic gradient".

Current Code:
```python
{current_code}
```

Feedback:
{error_summary}

Task: Extract a concise, actionable gradient description that captures:
1. What is wrong (the error pattern)
2. Where to fix (the scope: local variable, loop logic, boundary condition, etc.)
3. Direction of change (increase/decrease, add/remove, refactor, etc.)

Output format (one sentence):
"Gradient: [Action] [Scope] because [Reason]"

Example:
"Gradient: Fix off-by-one error in loop condition because index exceeds array bounds."

Your gradient:"""

        gradient = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=0.3  # Low temperature for consistent gradient extraction
        )
        
        # Clean up the response
        gradient = gradient.strip()
        if gradient.startswith("Gradient:"):
            gradient = gradient[9:].strip()
        
        self.gradient_history.append(gradient)
        return gradient
    
    def update_momentum(self, new_gradient: str) -> str:
        """
        Update momentum using exponential moving average.
        
        This is Step 2: combine the old momentum with the new gradient.
        
        Mathematical analogy:
            m_{t+1} = β * Summarize(m_t) + (1-β) * g_t
        
        Args:
            new_gradient: Current gradient from extract_gradient()
            
        Returns:
            Updated momentum text
        """
        if self.beta == 0.0:
            # No momentum: just use the current gradient
            self.momentum_text = new_gradient
            return self.momentum_text
        
        prompt = f"""You are updating the optimization momentum for a code repair process.

Previous Momentum (weight={self.beta:.1f}):
{self.momentum_text}

New Gradient (weight={1-self.beta:.1f}):
{new_gradient}

Task: Combine these into a single, coherent momentum statement that:
1. Preserves the overall direction from previous momentum (if consistent)
2. Incorporates new information from the gradient
3. Resolves conflicts by favoring the more specific/actionable direction
4. Stays concise (1-2 sentences max)

Output format:
"Momentum: [Overall optimization direction and focus areas]"

Example:
"Momentum: Focus on fixing loop boundary conditions. Maintain the current variable naming structure."

Your updated momentum:"""

        updated_momentum = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=0.5  # Moderate temperature for creative synthesis
        )
        
        # Clean up
        updated_momentum = updated_momentum.strip()
        if updated_momentum.startswith("Momentum:"):
            updated_momentum = updated_momentum[9:].strip()
        
        self.momentum_text = updated_momentum
        return self.momentum_text
    
    def step(
        self,
        task_description: str,
        current_code: str,
        feedback: Dict[str, Any]
    ) -> str:
        """
        Perform one optimization step with momentum.
        
        This is the main entry point that combines all components:
        1. Extract gradient from feedback
        2. Update momentum
        3. Generate new code with momentum constraint
        
        Args:
            task_description: Original task description
            current_code: Current code state
            feedback: Test results and errors
            
        Returns:
            New code after optimization step
        """
        # Step 1: Extract semantic gradient
        gradient = self.extract_gradient(current_code, feedback)
        
        # Step 2: Update momentum
        momentum = self.update_momentum(gradient)
        
        # Step 3: Generate new code with momentum constraint
        new_code = self._generate_with_momentum(
            task_description,
            current_code,
            feedback,
            momentum
        )
        
        # Record history
        self.code_history.append(new_code)
        if 'passed_tests' in feedback and 'total_tests' in feedback:
            loss = 1.0 - (feedback['passed_tests'] / feedback['total_tests'])
            self.loss_history.append(loss)
        
        return new_code
    
    def _generate_with_momentum(
        self,
        task_description: str,
        current_code: str,
        feedback: Dict[str, Any],
        momentum: str
    ) -> str:
        """
        Generate new code with momentum as a hard constraint.
        
        The momentum is injected into the system prompt to enforce consistency.
        """
        # Build context from recent history
        context = self._build_context()
        
        # Format feedback
        error_summary = self._format_feedback(feedback)
        
        # Construct prompt with momentum constraint
        system_prompt = f"""You are a code optimization agent with MOMENTUM CONTROL.

CRITICAL CONSTRAINT - Optimization Momentum:
{momentum}

You MUST:
1. Align all edits with this momentum direction
2. Do NOT introduce changes that contradict the momentum
3. Do NOT revert structural changes from previous steps
4. Focus on incremental improvements in the momentum direction

If the momentum conflicts with the feedback, prioritize fixing critical errors first,
but do so in a way that respects the overall optimization direction."""

        full_prompt = f"""{system_prompt}

Task:
{task_description}

Current Code:
```python
{current_code}
```

Test Feedback:
{error_summary}

{context}

Generate the improved code (output ONLY the code, no explanations):"""

        new_code = self.llm.generate_code(
            task_description=full_prompt,
            current_code="",
            error_feedback=None,
            temperature=self.temperature
        )
        
        return new_code
    
    def _build_context(self) -> str:
        """Build context from recent history."""
        if not self.gradient_history:
            return ""
        
        # Show last few gradients for reference
        recent_gradients = self.gradient_history[-self.max_history:]
        context = "\nRecent Optimization History:\n"
        for i, grad in enumerate(recent_gradients, 1):
            context += f"  Step -{len(recent_gradients)-i}: {grad}\n"
        
        return context
    
    def _format_feedback(self, feedback: Dict[str, Any]) -> str:
        """Format feedback into readable text."""
        if not feedback:
            return "No feedback available."
        
        parts = []
        
        if 'passed_tests' in feedback and 'total_tests' in feedback:
            parts.append(f"Tests: {feedback['passed_tests']}/{feedback['total_tests']} passed")
        
        if 'errors' in feedback and feedback['errors']:
            parts.append("\nErrors:")
            for i, error in enumerate(feedback['errors'][:3], 1):
                parts.append(f"  {i}. {error}")
        
        if 'has_syntax_error' in feedback and feedback['has_syntax_error']:
            parts.append("\n⚠ Syntax error detected - code cannot run")
        
        return '\n'.join(parts) if parts else "All tests passed!"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics for analysis."""
        return {
            'total_steps': len(self.code_history),
            'momentum_history': self.gradient_history,
            'loss_history': self.loss_history,
            'final_momentum': self.momentum_text,
            'beta': self.beta
        }
    
    def reset(self):
        """Reset optimizer state."""
        self.momentum_text = "No momentum yet. Starting fresh optimization."
        self.gradient_history = []
        self.code_history = []
        self.loss_history = []


class BaselineAgent:
    """
    Baseline agent without momentum (for comparison).
    
    This is equivalent to standard ReAct or Full-Context approaches.
    """
    
    def __init__(
        self,
        llm_kernel: LLMKernel,
        temperature: float = 0.0,
        max_history: int = 10
    ):
        self.llm = llm_kernel
        self.temperature = temperature
        self.max_history = max_history
        self.code_history: List[str] = []
        self.loss_history: List[float] = []
    
    def step(
        self,
        task_description: str,
        current_code: str,
        feedback: Dict[str, Any]
    ) -> str:
        """Standard optimization step without momentum."""
        # Format feedback
        error_summary = self._format_feedback(feedback)
        
        # Build context from full history
        context = self._build_context()
        
        prompt = f"""Task:
{task_description}

Current Code:
```python
{current_code}
```

Test Feedback:
{error_summary}

{context}

Generate the improved code (output ONLY the code, no explanations):"""

        new_code = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=self.temperature
        )
        
        # Record history
        self.code_history.append(new_code)
        if 'passed_tests' in feedback and 'total_tests' in feedback:
            loss = 1.0 - (feedback['passed_tests'] / feedback['total_tests'])
            self.loss_history.append(loss)
        
        return new_code
    
    def _build_context(self) -> str:
        """Build context from recent history."""
        if len(self.code_history) <= 1:
            return ""
        
        recent = self.code_history[-self.max_history:]
        return f"\nPrevious attempts: {len(recent)} iterations"
    
    def _format_feedback(self, feedback: Dict[str, Any]) -> str:
        """Format feedback into readable text."""
        if not feedback:
            return "No feedback available."
        
        parts = []
        
        if 'passed_tests' in feedback and 'total_tests' in feedback:
            parts.append(f"Tests: {feedback['passed_tests']}/{feedback['total_tests']} passed")
        
        if 'errors' in feedback and feedback['errors']:
            parts.append("\nErrors:")
            for i, error in enumerate(feedback['errors'][:3], 1):
                parts.append(f"  {i}. {error}")
        
        return '\n'.join(parts) if parts else "All tests passed!"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'total_steps': len(self.code_history),
            'loss_history': self.loss_history
        }
    
    def reset(self):
        """Reset agent state."""
        self.code_history = []
        self.loss_history = []
