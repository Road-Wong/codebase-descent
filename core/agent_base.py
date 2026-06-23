"""
Abstract Agent for Code Optimization.

The Agent is the Controller in the control-theory analogy:
    Action = f(Observation, Memory)

It observes the environment through Observations only (no direct
file access, no ground truth leakage) and produces Actions (patches).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import Action, Observation, SemanticGradient


class AbstractAgent(ABC):
    """
    Base class for all optimization agents.

    The agent's interface is minimal:
        - act(observation) → Action
        - reset()

    Subclasses implement different optimization strategies:
        - BaselineAgent: standard ReAct (no momentum)
        - SemanticMomentumOptimizer: SMO with EMA momentum
        - AdaptiveOptimizer: Adam-like adaptive temperature
    """

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Produce an Action (patch) given the current Observation.

        This is the core policy function π(o_t) → a_t.

        Args:
            observation: What the agent can see (code, errors, loss).

        Returns:
            Action containing a patch to apply.
        """
        ...

    def reset(self) -> None:
        """Reset agent state between episodes."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Return optimization statistics for analysis."""
        return {}
