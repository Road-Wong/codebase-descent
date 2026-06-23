"""
RL Episode Runner — the reconciliation loop.

This is the central orchestrator that drives the optimization dynamics:

    for t in range(max_steps):
        action   = agent.act(observation)        # Controller
        obs, r, done, info = env.step(action)    # Plant
        logger.record(step_t)                    # Observer

Dependency Injection:  Env and Agent are injected, not imported.
This allows seamless swapping of BaselineAgent ↔ SemanticMomentumOptimizer,
ConvexEnvironment ↔ SaddleEnvironment, etc.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from core.types import (
    Action,
    DynamicsStatus,
    EpisodeResult,
    ExperimentConfig,
    Observation,
    State,
    TrajectoryStep,
)


# ---------------------------------------------------------------------------
# Protocols (structural typing for DI)
# ---------------------------------------------------------------------------

class EnvironmentProtocol(Protocol):
    """Anything that looks like an environment."""
    def reset(self, initial_code: Optional[str] = None, seed: Optional[int] = None) -> Observation: ...
    def step(self, action: Action) -> tuple: ...
    @property
    def state(self) -> State: ...
    @property
    def is_done(self) -> bool: ...


class AgentProtocol(Protocol):
    """Anything that looks like an agent."""
    def act(self, observation: Observation) -> Action: ...
    def reset(self) -> None: ...


class AuditorProtocol(Protocol):
    """Anything that audits dynamics."""
    def step(self, loss: float, code: str) -> Any: ...
    def get_statistics(self) -> Dict[str, Any]: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# LoopRunner — the main engine
# ---------------------------------------------------------------------------

class LoopRunner:
    """
    Drives one RL episode: Agent interacts with Environment under
    the supervision of the Auditor.

    Usage:
        runner = LoopRunner(env, agent, auditor, logger)
        result = runner.run(initial_code="...", task_description="...")
    """

    def __init__(
        self,
        env: EnvironmentProtocol,
        agent: AgentProtocol,
        auditor: Optional[AuditorProtocol] = None,
        logger: Optional["TrajectoryLogger"] = None,
        max_steps: int = 20,
    ):
        self.env = env
        self.agent = agent
        self.auditor = auditor
        self.logger = logger
        self.max_steps = max_steps

    def run(
        self,
        initial_code: Optional[str] = None,
        task_description: str = "",
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        """
        Run one complete episode.

        Args:
            initial_code: Starting code (if None, env uses ground truth).
            task_description: Task spec for the agent.
            seed: Random seed.

        Returns:
            EpisodeResult with trajectory and statistics.
        """
        t_start = time.time()

        # Reset
        self.agent.reset()
        if self.auditor:
            self.auditor.reset()
        trajectory: List[TrajectoryStep] = []

        # Initial observation
        obs = self.env.reset(initial_code=initial_code, seed=seed)

        # Override task description if provided
        if task_description:
            obs = obs.model_copy(update={"task_description": task_description})

        for step in range(self.max_steps):
            # Agent acts
            action = self.agent.act(obs)

            # Environment transitions
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Auditor classifies dynamics
            dynamics_status = DynamicsStatus.WARMUP
            if self.auditor:
                dynamics_status = self.auditor.step(
                    next_obs.loss, next_obs.current_code
                )

            # Record trajectory step
            traj_step = TrajectoryStep(
                step=step,
                state=self.env.state,
                action=action,
                observation=next_obs,
                reward=reward,
                done=done,
                dynamics_status=dynamics_status,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            trajectory.append(traj_step)

            # Log to JSONL
            if self.logger:
                self.logger.record(traj_step)

            # Advance
            obs = next_obs

            if done or truncated:
                break

        # Collect statistics
        wall_time = time.time() - t_start
        auditor_stats = self.auditor.get_statistics() if self.auditor else {}
        agent_stats = self._get_agent_stats()

        return EpisodeResult(
            success=done,
            total_steps=step + 1,
            final_loss=obs.loss,
            min_loss=min(ts.observation.loss for ts in trajectory) if trajectory else 1.0,
            oscillation_index=auditor_stats.get("oscillation_index", 0.0),
            trajectory=trajectory,
            auditor_stats=auditor_stats,
            agent_stats=agent_stats,
            wall_time_seconds=wall_time,
        )

    def _get_agent_stats(self) -> Dict[str, Any]:
        """Extract stats from agent, if available."""
        if hasattr(self.agent, "get_statistics"):
            return self.agent.get_statistics()
        return {}


# ---------------------------------------------------------------------------
# TrajectoryLogger — JSONL phase space recorder
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """
    Records every step of the trajectory as a JSONL file.

    Each line is a complete TrajectoryStep serialized to JSON.
    This enables post-hoc analysis of the full phase space trajectory.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Clear file
        with open(output_path, "w") as f:
            pass

    def record(self, step: TrajectoryStep) -> None:
        """Append one step to the JSONL file."""
        with open(self.output_path, "a") as f:
            f.write(step.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Factory: build from ExperimentConfig
# ---------------------------------------------------------------------------

def create_env_from_config(config: ExperimentConfig) -> EnvironmentProtocol:
    """
    Factory: instantiate an Environment from ExperimentConfig.

    Maps config.environment.name → concrete environment class.
    """
    from environments import (
        ConvexEnvironment,
        SaddleEnvironment,
        LRUCacheEnvironment,
        GraphEnvironment,
        OrderDiscountEnvironment,
    )

    env_map = {
        "convex": ConvexEnvironment,
        "saddle": SaddleEnvironment,
        "lru_cache": LRUCacheEnvironment,
        "graph": GraphEnvironment,
        "order_discount": OrderDiscountEnvironment,
    }

    name = config.environment.name
    if name not in env_map:
        raise ValueError(f"Unknown environment: {name}. Choose from {list(env_map.keys())}")

    env_cls = env_map[name]
    return env_cls(obfuscation_level=config.environment.obfuscation_level)


def create_agent_from_config(
    config: ExperimentConfig, llm_kernel: Any = None
) -> AgentProtocol:
    """
    Factory: instantiate an Agent from ExperimentConfig.

    Maps config.agent.type → concrete agent class.
    """
    from agents import (
        SemanticMomentumOptimizer,
        BaselineAgent,
    )

    agent_type = config.agent.type
    if agent_type == "momentum":
        return SemanticMomentumOptimizer(
            llm_kernel=llm_kernel,
            beta=config.agent.beta,
            temperature=config.agent.temperature,
            max_history=config.agent.max_history,
        )
    elif agent_type == "baseline":
        return BaselineAgent(
            llm_kernel=llm_kernel,
            temperature=config.agent.temperature,
            max_history=config.agent.max_history,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_experiment(config: ExperimentConfig, llm_kernel: Any = None) -> List[EpisodeResult]:
    """
    Top-level entry point: run a full experiment.

    Creates Env, Agent, Auditor, Logger from config, then runs
    num_trials episodes.
    """
    from harness.auditor import DynamicsAuditor

    env = create_env_from_config(config)
    agent = create_agent_from_config(config, llm_kernel)
    auditor = DynamicsAuditor()

    results = []
    for trial in range(config.num_trials):
        logger = TrajectoryLogger(
            os.path.join(config.output_dir, f"trajectory_{trial}.jsonl")
        )
        runner = LoopRunner(
            env=env,
            agent=agent,
            auditor=auditor,
            logger=logger,
            max_steps=config.environment.max_steps,
        )
        result = runner.run(seed=config.seed)
        results.append(result)

    return results
