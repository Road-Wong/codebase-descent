"""
Core type definitions for the Codebase Descent RL framework.

Every data structure crossing module boundaries is a Pydantic model.
This enforces the "typed phase space" contract: State, Action, Observation
are the only allowed carriers of information between Env, Agent, and Harness.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. State — the full internal state of the code being optimized
# ---------------------------------------------------------------------------

class State(BaseModel):
    """
    The code manifold point c_t ∈ C.

    This is the *environment's* internal representation.  The Agent never
    sees State directly — it receives an Observation instead.
    """
    code: str = Field(..., description="Current source code snapshot c_t")
    step: int = Field(0, ge=0, description="Optimization step counter t")
    loss: float = Field(1.0, ge=0.0, le=1.0, description="Loss L(c_t) ∈ [0,1]")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Action — the gradient step Δc produced by the Agent
# ---------------------------------------------------------------------------

class ActionFormat(str, Enum):
    """Supported patch formats."""
    UNIFIED_DIFF = "unified_diff"      # standard diff -u
    SEARCH_REPLACE = "search_replace"  # search/replace blocks
    FULL_CODE = "full_code"            # legacy: full code replacement


class Action(BaseModel):
    """
    The agent's output: a patch Δc to apply to the current code.

    In the optimization analogy, this is the negative gradient step:
        c_{t+1} = c_t + Δc      (where Δc is a diff/patch)

    The Harness (loop_runner) is responsible for applying the patch to
    produce the next State.  The Agent never writes files or touches
    the environment directly.
    """
    patch: str = Field(..., description="The patch content (diff or full code)")
    format: ActionFormat = Field(ActionFormat.UNIFIED_DIFF, description="Patch format")
    confidence: float = Field(0.5, ge=0.0, le=1.0,
                              description="Agent's self-assessed confidence in this action")
    reasoning: Optional[str] = Field(None, description="Optional chain-of-thought trace")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="Extra data (e.g. _target_code for reliable patch application)")


# ---------------------------------------------------------------------------
# 3. Observation — what the Agent is allowed to see
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    The Agent's *only* view into the environment.

    Enforces information asymmetry: the Agent cannot read files, cannot
    inspect the environment's ground truth, and cannot bypass the test
    harness.  It sees exactly:
      - current_code:  the code snapshot c_t
      - test_result:   pass/fail count + error traces
      - loss:          scalar loss value (the reward signal)
      - task_description: what it is trying to implement
      - momentum:      optional semantic momentum text (SMO only)
      - recent_gradients: optional gradient history (SMO only)

    This is the "observation function" O: State → Observation in the
    POMDP formulation — the Agent operates under partial observability.
    """
    current_code: str = Field(..., description="Current code snapshot c_t")
    task_description: str = Field(..., description="What the agent is trying to implement")
    loss: float = Field(..., ge=0.0, le=1.0, description="Current loss L(c_t)")
    passed_tests: int = Field(0, ge=0)
    total_tests: int = Field(0, ge=0)
    errors: List[str] = Field(default_factory=list, description="Test failure traces")
    has_syntax_error: bool = Field(False)
    step: int = Field(0, ge=0, description="Current optimization step t")

    # SMO-specific fields (None for BaselineAgent)
    momentum: Optional[str] = Field(None, description="Semantic momentum text")
    recent_gradients: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. SemanticGradient — structured gradient extracted by SMO
# ---------------------------------------------------------------------------

class SemanticGradient(BaseModel):
    """
    The "gradient" in natural language space, extracted by SMO's
    extract_gradient() step.

    Analogous to ∇L(c_t) in continuous optimization, but expressed as
    structured natural language describing what needs to change.
    """
    action: str = Field(..., description="What to do (e.g. 'Fix off-by-one')")
    scope: str = Field(..., description="Where to fix (e.g. 'loop condition')")
    reason: str = Field(..., description="Why (e.g. 'index exceeds bounds')")
    raw_text: str = Field(..., description="Full gradient text from LLM")


# ---------------------------------------------------------------------------
# 5. TrajectoryStep — one point in the phase space trajectory
# ---------------------------------------------------------------------------

class DynamicsStatus(str, Enum):
    """Trajectory classification (from DynamicsAuditor)."""
    WARMUP = "Warming Up"
    CONVERGING = "Normal Descent"
    OSCILLATING = "Limit Cycle Detected"
    STAGNATING = "Vanishing Gradient"
    DIVERGING = "Exploding Gradient"
    SOLVED = "Converged to Solution"


class TrajectoryStep(BaseModel):
    """
    One step in the phase space trajectory, recorded by the logger.

    This is the "complete measurement" at time t: state, action taken,
    reward received, and the auditor's classification of the dynamics.
    """
    step: int = Field(..., ge=0)
    state: State
    action: Action
    observation: Observation
    reward: float = Field(..., description="=-loss (RL convention: higher is better)")
    done: bool = Field(False)
    dynamics_status: DynamicsStatus = Field(DynamicsStatus.WARMUP)
    timestamp: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# 6. ExperimentConfig — typed hyperparameter bundle
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Agent hyperparameters."""
    type: str = Field("baseline", description="Agent type: 'baseline' or 'momentum'")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature η")
    beta: float = Field(0.7, ge=0.0, le=1.0, description="Momentum coefficient β (SMO only)")
    max_history: int = Field(3, ge=1, description="Context window size K")


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    name: str = Field("saddle", description="Environment name")
    obfuscation_level: int = Field(0, ge=0, le=3, description="De-semanticization level")
    max_steps: int = Field(20, ge=1, description="Max steps per episode")


class ExperimentConfig(BaseModel):
    """Full experiment configuration (loaded from YAML)."""
    experiment_name: str = Field("default")
    num_trials: int = Field(1, ge=1)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    seed: Optional[int] = Field(None)
    output_dir: str = Field("outputs")


class EpisodeResult(BaseModel):
    """Complete result of one RL episode."""
    success: bool
    total_steps: int
    final_loss: float
    min_loss: float
    oscillation_index: float
    trajectory: List[TrajectoryStep]
    auditor_stats: Dict[str, Any] = Field(default_factory=dict)
    agent_stats: Dict[str, Any] = Field(default_factory=dict)
    wall_time_seconds: float = 0.0
