"""
Tests for Pydantic type definitions (from test_new_architecture.py).
"""

from core.types import (
    State, Action, Observation, SemanticGradient,
    TrajectoryStep, DynamicsStatus, ActionFormat,
    ExperimentConfig, AgentConfig, EnvironmentConfig,
)


def test_state():
    s = State(code="def f(): pass", step=0, loss=1.0)
    assert s.code == "def f(): pass"
    assert s.step == 0


def test_action():
    a = Action(patch="--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new", format=ActionFormat.UNIFIED_DIFF)
    assert a.format == ActionFormat.UNIFIED_DIFF


def test_observation():
    o = Observation(
        current_code="def f(): pass",
        task_description="Implement f",
        loss=0.5,
        passed_tests=3,
        total_tests=6,
    )
    assert o.loss == 0.5
    assert o.passed_tests == 3


def test_experiment_config():
    cfg = ExperimentConfig(
        experiment_name="test",
        num_trials=1,
        agent=AgentConfig(type="momentum", temperature=0.7, beta=0.7),
        environment=EnvironmentConfig(name="saddle", obfuscation_level=1),
    )
    assert cfg.agent.type == "momentum"
    assert cfg.environment.name == "saddle"
