"""
Tests for experiment logger (from test_new_architecture.py).
"""

import json
from harness.logger import TrajectoryLogger, ExperimentLogger
from core.types import (
    TrajectoryStep, State, Action, Observation, DynamicsStatus, ActionFormat,
)


def test_trajectory_logger(output_dir="outputs"):
    logger = TrajectoryLogger(f"{output_dir}/test_traj_logger.jsonl")

    step = TrajectoryStep(
        step=0,
        state=State(code="def f(): pass", step=0, loss=1.0),
        action=Action(patch="def f(): return 1", format=ActionFormat.FULL_CODE),
        observation=Observation(
            current_code="def f(): return 1",
            task_description="test",
            loss=0.5,
            passed_tests=1,
            total_tests=2,
        ),
        reward=-0.5,
        done=False,
        dynamics_status=DynamicsStatus.CONVERGING,
    )
    logger.record(step)

    with open(f"{output_dir}/test_traj_logger.jsonl", "r") as f:
        data = json.loads(f.readline())
    assert data["step"] == 0
    assert data["reward"] == -0.5
