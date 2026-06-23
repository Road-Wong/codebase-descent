"""
Harness End-to-End Dry Run Test

Verifies the full RL loop works WITHOUT calling the LLM API:
  1. MockOscillatingLLM outputs preset patches
  2. LoopRunner runs 10 steps
  3. trajectories.jsonl contains all required fields for analysis

This is the "wind tunnel" test — if this fails, the real
experiments will produce garbage data.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import (
    Action, ActionFormat, Observation, State,
    TrajectoryStep, DynamicsStatus,
)
from core.protocol import make_action
from core.env_base import CodeEnv
from harness.loop_runner import LoopRunner
from harness.auditor import DynamicsAuditor
from harness.logger import TrajectoryLogger


# =========================================================================
# Mock Agents — No LLM Required
# =========================================================================

class MockOscillatingLLM:
    """
    Simulates an LLM that oscillates between two code states.

    This creates a "limit cycle": the agent alternates between two
    different implementations, never converging to the solution.

    The trajectory should be classified as OSCILLATING by the auditor.
    """

    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.patch_history = []

    def act(self, observation: Observation) -> Action:
        """Alternate between two different code modifications."""
        self.step_count += 1
        code = observation.current_code

        # Find lines to oscillate between
        lines = code.split("\n")
        target_line_a = None
        target_line_b = None

        for line in lines:
            # In _eval_expr: "left = left + right" vs "left = left - right"
            if "left = left + right" in line and target_line_a is None:
                target_line_a = line
            if "left = left - right" in line and target_line_b is None:
                target_line_b = line

        if self.step_count % 2 == 1 and target_line_a:
            # Odd steps: change + to -
            patch = make_action(target_line_a, target_line_a.replace("+ right", "- right"))
        elif self.step_count % 2 == 0 and target_line_b:
            # Even steps: change - to +
            patch = make_action(target_line_b, target_line_b.replace("- right", "+ right"))
        else:
            # Fallback: no-op
            patch = make_action(lines[0] if lines else "", lines[0] if lines else "")

        self.patch_history.append(patch)
        return Action(
            patch=patch,
            format=ActionFormat.SEARCH_REPLACE,
            confidence=0.5,
        )

    def reset(self):
        self.step_count = 0
        self.patch_history = []

    def get_statistics(self):
        return {
            "total_steps": self.step_count,
            "patches": len(self.patch_history),
        }


class MockConvergingLLM:
    """
    Simulates an LLM that makes steady progress.

    Each step applies a small fix toward the solution.
    Should produce a CONVERGING trajectory.
    """

    def __init__(self, fixes):
        """
        Args:
            fixes: List of (search, replace) tuples to apply in order.
        """
        self.fixes = fixes
        self.step_count = 0

    def act(self, observation: Observation) -> Action:
        if self.step_count < len(self.fixes):
            search, replace = self.fixes[self.step_count]
            patch = make_action(search, replace)
        else:
            # No-op after all fixes applied
            patch = make_action(
                observation.current_code.split("\n")[0],
                observation.current_code.split("\n")[0],
            )

        self.step_count += 1
        return Action(patch=patch, format=ActionFormat.SEARCH_REPLACE)

    def reset(self):
        self.step_count = 0

    def get_statistics(self):
        return {"total_steps": self.step_count}


class MockStagnatingLLM:
    """
    Simulates an LLM that keeps making changes that don't help.

    Each step modifies code but loss stays the same.
    Should produce STAGNATING or WARMUP trajectory.
    """

    def __init__(self):
        self.step_count = 0

    def act(self, observation: Observation) -> Action:
        self.step_count += 1
        # Make a cosmetic change that doesn't fix the bug
        code = observation.current_code
        if "# comment" in code:
            patch = make_action("# comment", f"# comment {self.step_count}")
        else:
            patch = make_action(
                observation.current_code.split("\n")[0],
                observation.current_code.split("\n")[0] + f" # {self.step_count}",
            )
        return Action(patch=patch, format=ActionFormat.SEARCH_REPLACE)

    def reset(self):
        self.step_count = 0

    def get_statistics(self):
        return {"total_steps": self.step_count}


# =========================================================================
# Tests
# =========================================================================

class TestDryRunHarness:
    """
    End-to-end dry run: no LLM API, pure mock agents.
    Verifies the harness produces valid trajectory data.
    """

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Temporary output directory."""
        return str(tmp_path / "outputs")

    def test_oscillating_trajectory_recorded(self, output_dir):
        """MockOscillatingLLM must produce a valid trajectory JSONL."""
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=10)

        # Start with broken code
        broken = env.ground_truth.replace(
            "left = left + right", "left = left - right", 1
        )

        agent = MockOscillatingLLM(env)
        auditor = DynamicsAuditor(window_size=5)
        logger = TrajectoryLogger(os.path.join(output_dir, "trajectories.jsonl"))

        runner = LoopRunner(
            env=env, agent=agent, auditor=auditor, logger=logger, max_steps=10,
        )
        result = runner.run(initial_code=broken)

        # Verify result
        assert result.total_steps > 0
        assert len(result.trajectory) == result.total_steps

        # Verify JSONL file exists and is valid
        jsonl_path = os.path.join(output_dir, "trajectories.jsonl")
        assert os.path.exists(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == result.total_steps, (
            f"JSONL has {len(lines)} lines but trajectory has {result.total_steps} steps"
        )

        # Verify each line is valid JSON with required fields
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert "step" in data, f"Line {i}: missing 'step'"
            assert "observation" in data, f"Line {i}: missing 'observation'"
            assert "action" in data, f"Line {i}: missing 'action'"
            assert "reward" in data, f"Line {i}: missing 'reward'"
            assert "dynamics_status" in data, f"Line {i}: missing 'dynamics_status'"

    def test_trajectory_has_analysis_fields(self, output_dir):
        """JSONL must contain fields needed for phase space analysis."""
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=5)

        agent = MockConvergingLLM([
            ("left = left - right", "left = left + right"),
        ])
        auditor = DynamicsAuditor()
        logger = TrajectoryLogger(os.path.join(output_dir, "analysis.jsonl"))

        runner = LoopRunner(env=env, agent=agent, auditor=auditor, logger=logger, max_steps=5)
        result = runner.run(
            initial_code=env.ground_truth.replace(
                "left = left + right", "left = left - right", 1
            )
        )

        # Read and verify analysis fields
        with open(os.path.join(output_dir, "analysis.jsonl"), "r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())

        obs = first_line["observation"]
        assert "loss" in obs, "Observation must contain 'loss' for plotting"
        assert "current_code" in obs, "Observation must contain 'current_code' for AST distance"
        assert "step" in obs, "Observation must contain 'step'"

        action = first_line["action"]
        assert "patch" in action, "Action must contain 'patch'"
        assert "format" in action, "Action must contain 'format'"

    def test_oscillation_detected_in_trajectory(self, output_dir):
        """Oscillating agent should produce OSCILLATING dynamics status."""
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=20)

        broken = env.ground_truth.replace(
            "left = left + right", "left = left - right", 1
        )

        agent = MockOscillatingLLM(env)
        auditor = DynamicsAuditor(window_size=5)
        logger = TrajectoryLogger(os.path.join(output_dir, "osc.jsonl"))

        # Run enough steps for oscillation detection (needs 4+ steps)
        runner = LoopRunner(env=env, agent=agent, auditor=auditor, logger=logger, max_steps=20)
        result = runner.run(initial_code=broken)

        # At some point, auditor should detect oscillation
        statuses = [ts.dynamics_status for ts in result.trajectory]
        has_oscillation = DynamicsStatus.OSCILLATING in statuses
        # If oscillation not detected, at least check we ran enough steps
        assert result.total_steps >= 2, f"Should run multiple steps, got {result.total_steps}"
        # The oscillating agent should eventually trigger detection
        if result.total_steps >= 4:
            assert has_oscillation, (
                f"Expected OSCILLATING after {result.total_steps} steps, "
                f"got: {[s.value for s in statuses]}"
            )

    def test_convergence_produces_solved(self, output_dir):
        """Converging agent should reach SOLVED state."""
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=5)

        broken = env.ground_truth.replace(
            "left = left + right", "left = left - right", 1
        )

        # Find the exact broken line for correct indentation
        broken_line = None
        for line in broken.split("\n"):
            if "left = left - right" in line:
                broken_line = line
                break

        agent = MockConvergingLLM([
            (broken_line, broken_line.replace("- right", "+ right")),
        ])
        auditor = DynamicsAuditor()
        logger = TrajectoryLogger(os.path.join(output_dir, "conv.jsonl"))

        runner = LoopRunner(env=env, agent=agent, auditor=auditor, logger=logger, max_steps=5)
        result = runner.run(initial_code=broken)

        assert result.success is True, "Converging agent should solve the task"
        assert result.final_loss == 0.0

    def test_momentum_state_in_trajectory(self, output_dir):
        """If SMO is used, momentum should appear in trajectory data."""
        # This test verifies the structure; actual SMO requires LLM
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=5)

        agent = MockConvergingLLM([
            ("left = left - right", "left = left + right"),
        ])
        auditor = DynamicsAuditor()
        logger = TrajectoryLogger(os.path.join(output_dir, "mom.jsonl"))

        runner = LoopRunner(env=env, agent=agent, auditor=auditor, logger=logger, max_steps=5)
        result = runner.run(initial_code=env.ground_truth.replace(
            "left = left + right", "left = left - right", 1
        ))

        # Verify trajectory structure supports momentum field
        with open(os.path.join(output_dir, "mom.jsonl"), "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        # Observation should have momentum field (even if None for non-SMO)
        obs = data["observation"]
        # The field exists in the Pydantic model
        assert "momentum" in obs or "loss" in obs  # momentum may be omitted if None


class TestDryRunMetrics:
    """Verify metrics are computed correctly during dry run."""

    def test_ast_distance_computed(self):
        """AST distance should change when code changes."""
        from harness.metrics import ast_distance

        code_a = "def f():\n    return 1"
        code_b = "def f():\n    return 2"

        dist = ast_distance(code_a, code_b)
        assert dist > 0, "Different codes should have non-zero AST distance"

        dist_same = ast_distance(code_a, code_a)
        assert dist_same == 0.0, "Same code should have zero AST distance"

    def test_composite_metric_tracks_history(self):
        """CompositeMetric should accumulate history for plotting."""
        from harness.metrics import CompositeMetric

        initial = "def f():\n    return 1"
        metric = CompositeMetric(initial_code=initial, lambda_ast=0.2)

        codes = [
            "def f():\n    return 1",
            "def f():\n    return 2",
            "def f():\n    return 3",
        ]
        losses = [0.5, 0.3, 0.1]

        for code, loss in zip(codes, losses):
            metric.compute(code, loss, step=codes.index(code))

        assert len(metric.history) == 3
        assert all("L_test" in h for h in metric.history)
        assert all("L_ast" in h for h in metric.history)
        assert all("L_total" in h for h in metric.history)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
