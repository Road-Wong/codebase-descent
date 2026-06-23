"""
Tests for DiffAgent (from test_sgd_smo.py).
"""

from agents.base_agent import DiffAgent
from core.types import Observation, ActionFormat
from core.protocol import validate_action


class GoodDiffAgent(DiffAgent):
    def _generate_patch(self, obs):
        return """<<<< SEARCH
old line
====
new line
>>>>"""


class BadDiffAgent(DiffAgent):
    def _generate_patch(self, obs):
        return "this is not a valid patch format"


def test_diff_agent_output_format():
    agent = GoodDiffAgent()
    obs = Observation(
        current_code="old line\nother line",
        task_description="test",
        loss=0.5,
        errors=["Test failed"],
    )
    action = agent.act(obs)

    assert action.format == ActionFormat.SEARCH_REPLACE
    is_valid, _ = validate_action(action.patch)
    assert is_valid


def test_diff_agent_rejects_bad_format():
    agent = BadDiffAgent()
    obs = Observation(
        current_code="some code",
        task_description="test",
        loss=0.5,
        errors=["error"],
    )
    action = agent.act(obs)

    assert action.confidence == 0.0
    assert "Format error" in (action.reasoning or "") or "ProtocolError" in (action.reasoning or "")
