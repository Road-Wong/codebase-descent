"""
Tests for full episode with diff-based actions (from test_new_architecture.py).
"""

import json
from environments import ConvexEnvironment
from harness.loop_runner import LoopRunner
from harness.auditor import DynamicsAuditor
from harness.logger import TrajectoryLogger
from core.types import Observation
from core.diff_utils import make_diff_action


class DiffAgent:
    """Agent that returns unified diffs."""
    def __init__(self, target_code):
        self.target_code = target_code
        self.steps = 0

    def act(self, observation: Observation):
        self.steps += 1
        return make_diff_action(
            old_code=observation.current_code,
            new_code=self.target_code,
            confidence=0.9,
        )

    def reset(self):
        self.steps = 0

    def get_statistics(self):
        return {"steps": self.steps}


def test_full_episode_with_diff(output_dir="outputs"):
    env = ConvexEnvironment(obfuscation_level=0, max_steps=5)
    agent = DiffAgent("def compute(x):\n    return 2 * x + 3")
    auditor = DynamicsAuditor()
    logger = TrajectoryLogger(f"{output_dir}/test_diff_episode.jsonl")

    runner = LoopRunner(env=env, agent=agent, auditor=auditor, logger=logger, max_steps=5)
    result = runner.run()

    assert result.success is True
    assert result.final_loss == 0.0
    assert len(result.trajectory) == 1

    with open(f"{output_dir}/test_diff_episode.jsonl", "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["action"]["format"] == "unified_diff"
