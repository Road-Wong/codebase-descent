"""
Semantic Momentum Optimizer (SMO)

The core academic contribution: a natural-language momentum vector that
stabilizes LLM code optimization against oscillation.

Mathematical formulation:
    m_{t+1} = β * m_t + (1-β) * g_t

Where:
    m_t : momentum vector (natural language summary of optimization direction)
    g_t : gradient (extracted from SINGLE error message + last diff)
    β   : momentum coefficient (0.0 = no memory, 0.9 = high inertia)

How it prevents LLM "cheating jumps":
  Without momentum, the LLM sees each error independently and may:
  - Fix error A by introducing error B (oscillation)
  - Revert a previous fix (limit cycle)
  - Ignore the trajectory and start from scratch each step (no learning)

  With momentum, the LLM receives a CONSTRAINT that says:
  "Your previous fixes established direction X.  Do NOT contradict X."
  This acts as a low-pass filter on the gradient signal, suppressing
  high-frequency oscillations (noise) while preserving the trend.

Gradient extraction from SGD feedback:
  The gradient g_t is NOT the full error report.  It is extracted from:
  1. The SINGLE error message from the SGD evaluator (stochastic sample)
  2. The LAST diff the agent applied (what changed)
  This mirrors SGD where each gradient is computed from ONE data point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_agent import DiffAgent
from .llm_kernel import LLMKernel
from core.types import Observation
from core.protocol import make_action, extract_protocol_block


class SemanticMomentumOptimizer(DiffAgent):
    """
    Semantic Momentum Optimizer — the core algorithm.

    Inherits from DiffAgent to enforce SEARCH/REPLACE output format.
    Maintains EMA momentum in natural language space.

    Three-step loop per optimization step:
      1. extract_gradient()  — LLM analyzes single error + last diff → g_t
      2. update_momentum()   — EMA: m_{t+1} = β*m_t + (1-β)*g_t
      3. generate_patch()    — LLM generates SEARCH/REPLACE with m_{t+1} as constraint
    """

    def __init__(
        self,
        llm_kernel: LLMKernel,
        beta: float = 0.7,
        temperature: float = 0.7,
        max_history: int = 3,
    ):
        super().__init__()
        self.llm = llm_kernel
        self.beta = beta
        self.temperature = temperature
        self.max_history = max_history

        # Momentum state (natural language)
        self.momentum_text = "No momentum yet. Starting fresh optimization."

        # History for gradient extraction
        self.gradient_history: List[str] = []
        self.diff_history: List[str] = []  # last applied diffs
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # DiffAgent interface: generate SEARCH/REPLACE patch
    # ------------------------------------------------------------------

    def _generate_patch(self, observation: Observation) -> str:
        """
        Main entry point: produce a SEARCH/REPLACE patch.

        Pipeline:
          1. Extract gradient from single error + last diff
          2. Update momentum via EMA
          3. Generate patch with momentum as hard constraint
        """
        # Step 1: Extract gradient from single SGD error
        gradient = self._extract_gradient(observation)

        # Step 2: EMA momentum update
        self._update_momentum(gradient)

        # Step 3: Generate SEARCH/REPLACE patch
        patch_text = self._generate_with_momentum(observation)

        # Record history
        self.diff_history.append(patch_text)
        if observation.loss is not None:
            self.loss_history.append(observation.loss)

        return patch_text

    # ------------------------------------------------------------------
    # Step 1: Gradient extraction from single error
    # ------------------------------------------------------------------

    def _extract_gradient(self, observation: Observation) -> str:
        """
        Extract semantic gradient g_t from the SINGLE error message.

        This is the "stochastic gradient" in our SGD analogy:
          - Full error report  → full-batch gradient (deterministic)
          - Single error       → single-sample gradient (stochastic)

        The LLM analyzes:
          1. The single error message (the only signal from the environment)
          2. The last diff we applied (what we changed)
          3. The current code (context)

        And extracts: "what went wrong, where, and which direction to fix."
        """
        error_msg = observation.errors[0] if observation.errors else "No error"
        last_diff = self.diff_history[-1] if self.diff_history else "No previous changes"

        prompt = f"""You are analyzing a code optimization step to extract a "semantic gradient".

CURRENT CODE:
```python
{observation.current_code[:2000]}
```

LAST CHANGE YOU MADE:
{last_diff[:500]}

SINGLE ERROR FROM TEST:
{error_msg}

TASK: Extract a concise gradient that captures:
1. What went wrong (the error pattern)
2. Where it went wrong (scope: which function/line)
3. Direction of fix (increase/decrease, add/remove, refactor)

Output EXACTLY one sentence in this format:
Gradient: [Action] [Scope] because [Reason]

Example:
Gradient: Fix off-by-one in _eval_term loop because index exceeds token list bounds.

Your gradient:"""

        gradient = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=0.3,  # Low temp for consistent gradient extraction
        )

        gradient = gradient.strip()
        if gradient.startswith("Gradient:"):
            gradient = gradient[9:].strip()

        self.gradient_history.append(gradient)
        return gradient

    # ------------------------------------------------------------------
    # Step 2: EMA momentum update
    # ------------------------------------------------------------------

    def _update_momentum(self, new_gradient: str) -> str:
        """
        Update momentum via Exponential Moving Average:

            m_{t+1} = β * m_t + (1-β) * g_t

        In continuous optimization, this averages out noisy gradients.
        In our natural-language space, the LLM "summarizes" the old
        momentum and the new gradient into a coherent direction.

        The β parameter controls inertia:
          β = 0.0  → no memory (pure SGD, oscillates)
          β = 0.7  → moderate memory (standard momentum)
          β = 0.9  → high memory (strong smoothing, slow to react)
        """
        if self.beta == 0.0:
            # No momentum: just use the current gradient
            self.momentum_text = new_gradient
            return self.momentum_text

        prompt = f"""You are updating the optimization momentum for a code repair process.

PREVIOUS MOMENTUM (weight={self.beta:.1f}):
{self.momentum_text}

NEW GRADIENT (weight={1-self.beta:.1f}):
{new_gradient}

TASK: Combine these into a single coherent momentum statement:
1. Preserve the overall direction from previous momentum (if consistent)
2. Incorporate new information from the gradient
3. If they conflict, favor the more specific/actionable direction
4. Stay concise (1-2 sentences max)

Output EXACTLY:
Momentum: [Combined optimization direction]

Your updated momentum:"""

        updated = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=0.5,  # Moderate temp for creative synthesis
        )

        updated = updated.strip()
        if updated.startswith("Momentum:"):
            updated = updated[9:].strip()

        self.momentum_text = updated
        return self.momentum_text

    # ------------------------------------------------------------------
    # Step 3: Patch generation with momentum constraint
    # ------------------------------------------------------------------

    def _generate_with_momentum(self, observation: Observation) -> str:
        """
        Generate SEARCH/REPLACE patch with momentum as hard constraint.

        The momentum is injected as a SYSTEM-LEVEL constraint with
        HIGHEST PRIORITY.  This forces the LLM to align its patch
        with the accumulated optimization direction.

        If the momentum says "focus on loop boundaries", the LLM
        cannot suddenly start refactoring variable names — it must
        address loop boundaries first.
        """
        error_msg = observation.errors[0] if observation.errors else "All tests passed"

        # Build recent gradient context
        recent_grads = self.gradient_history[-self.max_history:]
        grad_context = "\n".join(
            f"  Step -{len(recent_grads)-i}: {g}"
            for i, g in enumerate(recent_grads)
        )

        prompt = f"""You are a code optimization agent with MOMENTUM CONTROL.

=== CRITICAL CONSTRAINT (HIGHEST PRIORITY) ===
Optimization Momentum:
{self.momentum_text}

YOU MUST:
1. Align ALL edits with this momentum direction
2. Do NOT introduce changes that contradict the momentum
3. Do NOT revert structural changes from previous steps
4. Focus on incremental improvements in the momentum direction
=== END CONSTRAINT ===

CURRENT CODE:
```python
{observation.current_code}
```

SINGLE ERROR FROM TEST:
{error_msg}

RECENT GRADIENT HISTORY:
{grad_context}

TASK: Generate a SEARCH/REPLACE patch to fix the error.
The patch MUST respect the momentum constraint above.

Output EXACTLY in this format:
<<<< SEARCH
[exact code to find]
====
[replacement code]
>>>>

Generate the patch now:"""

        patch_text = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=self.temperature,
            raw=True,
            system_prompt="You are a code optimization agent. You MUST output SEARCH/REPLACE patches in the exact format requested. Never output raw code blocks — always use <<<< SEARCH / ==== / >>>> markers. Output ONLY the patch, no explanations.",
        )

        # Extract protocol block from response (strip any preamble/explanation)
        patch_text = extract_protocol_block(patch_text.strip())

        if "<<<< SEARCH" not in patch_text:
            if "```python" in patch_text:
                start = patch_text.find("```python") + 9
                end = patch_text.find("```", start)
                if end != -1:
                    code_block = patch_text[start:end].strip()
                    lines = observation.current_code.strip().split("\n")
                    search_line = lines[0] if lines else ""
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("class "):
                            search_line = line
                            break
                    return make_action(search_line, code_block)
            lines = observation.current_code.strip().split("\n")
            return make_action(lines[0] if lines else "", lines[0] if lines else "")

        return patch_text

    # ------------------------------------------------------------------
    # Reset and statistics
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset optimizer state between episodes."""
        super().reset()
        self.momentum_text = "No momentum yet. Starting fresh optimization."
        self.gradient_history = []
        self.diff_history = []
        self.loss_history = []

    def get_statistics(self) -> Dict[str, Any]:
        """Return optimization statistics including momentum state."""
        base = super().get_statistics()
        base.update({
            "momentum_text": self.momentum_text,
            "gradient_history": self.gradient_history[-self.max_history:],
            "loss_history": self.loss_history,
            "beta": self.beta,
        })
        return base


# ---------------------------------------------------------------------------
# Baseline Agent (no momentum) — for comparison experiments
# ---------------------------------------------------------------------------

class BaselineAgent(DiffAgent):
    """
    Baseline agent without momentum (control group).

    Equivalent to standard ReAct: each step is independent.
    No memory of previous gradients, no directional constraint.

    This is the "SGD without momentum" baseline:
      m_t = g_t  (β = 0, no smoothing)

    Expected behavior:
      - Oscillates more than SMO
      - Reverts previous fixes
      - Higher oscillation index
    """

    def __init__(self, llm_kernel: LLMKernel, temperature: float = 0.7):
        super().__init__()
        self.llm = llm_kernel
        self.temperature = temperature

    def _generate_patch(self, observation: Observation) -> str:
        """Generate patch without momentum — pure reactive."""
        error_msg = observation.errors[0] if observation.errors else "All tests passed"

        prompt = f"""You are a code optimization agent.

CURRENT CODE:
```python
{observation.current_code}
```

ERROR FROM TEST:
{error_msg}

TASK: Generate a SEARCH/REPLACE patch to fix the error.

Output EXACTLY in this format:
<<<< SEARCH
[exact code to find]
====
[replacement code]
>>>>

Generate the patch now:"""

        patch_text = self.llm.generate_code(
            task_description=prompt,
            current_code="",
            error_feedback=None,
            temperature=self.temperature,
            raw=True,
            system_prompt="You are a code optimization agent. You MUST output SEARCH/REPLACE patches in the exact format requested. Never output raw code blocks — always use <<<< SEARCH / ==== / >>>> markers. Output ONLY the patch, no explanations.",
        )

        # Extract protocol block from response (strip any preamble/explanation)
        patch_text = extract_protocol_block(patch_text.strip())

        if "<<<< SEARCH" not in patch_text:
            if "```python" in patch_text:
                start = patch_text.find("```python") + 9
                end = patch_text.find("```", start)
                if end != -1:
                    code_block = patch_text[start:end].strip()
                    lines = observation.current_code.strip().split("\n")
                    search_line = lines[0] if lines else ""
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("class "):
                            search_line = line
                            break
                    return make_action(search_line, code_block)
            lines = observation.current_code.strip().split("\n")
            return make_action(lines[0] if lines else "", lines[0] if lines else "")

        return patch_text
