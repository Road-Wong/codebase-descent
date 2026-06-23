"""
Tests for diff-based actions (from test_new_architecture.py).
"""

from environments import ConvexEnvironment
from core.diff_utils import generate_diff, apply_diff, make_diff_action
from core.types import Action, ActionFormat


def test_diff_generation():
    old_code = "def f():\n    return 1"
    new_code = "def f():\n    return 2"
    diff = generate_diff(old_code, new_code)
    assert "---" in diff
    assert "+++" in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_diff_application():
    old_code = "def f():\n    return 1"
    new_code = "def f():\n    return 2"
    diff = generate_diff(old_code, new_code)
    applied = apply_diff(old_code, diff)
    assert applied == new_code


def test_make_diff_action():
    old_code = "def f():\n    return 1"
    new_code = "def f():\n    return 2"
    action = make_diff_action(old_code, new_code, confidence=0.8)
    assert action.format == ActionFormat.UNIFIED_DIFF
    assert action.confidence == 0.8
    assert action.metadata.get("_target_code") == new_code


def test_env_with_diff_action():
    env = ConvexEnvironment(obfuscation_level=0, max_steps=10)
    obs = env.reset()

    correct_code = "def compute(x):\n    return 2 * x + 3"
    action = make_diff_action(obs.current_code, correct_code)
    obs, reward, done, truncated, info = env.step(action)

    assert done is True
    assert obs.loss == 0.0
