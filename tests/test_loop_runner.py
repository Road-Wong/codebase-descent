"""
Tests for LoopRunner episode orchestration (from test_new_architecture.py).
"""

from environments import ConvexEnvironment
from harness.loop_runner import LoopRunner, TrajectoryLogger
from harness.auditor import DynamicsAuditor
from core.types import Action, ActionFormat, Observation


class FixedAgent:
    """Agent that always returns the same correct code."""
    def __init__(self, code):
        self.code = code
        self.call_count = 0

    def act(self, observation: Observation) -> Action:
        self.call_count += 1
        return Action(patch=self.code, format=ActionFormat.FULL_CODE)

    def reset(self):
        self.call_count = 0

    def get_statistics(self):
        return {"calls": self.call_count}


def test_loop_runner_solves_convex(output_dir="outputs"):
    env = ConvexEnvironment(obfuscation_level=0, max_steps=5)
    agent = FixedAgent("def compute(x):\n    return 2 * x + 3")
    auditor = DynamicsAuditor()
    logger = TrajectoryLogger(f"{output_dir}/test_loop_runner.jsonl")

    runner = LoopRunner(env=env, agent=agent, auditor=auditor, logger=logger, max_steps=5)
    result = runner.run()

    assert result.success is True
    assert result.total_steps == 1
    assert result.final_loss == 0.0
    assert len(result.trajectory) == 1
