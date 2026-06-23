"""
Tests for environment Gymnasium interface (from test_new_architecture.py).
"""

from environments import ConvexEnvironment, SaddleEnvironment
from core.types import Action, ActionFormat


def test_convex_reset():
    env = ConvexEnvironment(obfuscation_level=0, max_steps=10)
    obs = env.reset()
    assert obs.loss >= 0.0
    assert obs.total_tests == 6
    assert obs.step == 0


def test_convex_step_correct_code():
    env = ConvexEnvironment(obfuscation_level=0, max_steps=10)
    env.reset()

    correct_code = "def compute(x):\n    return 2 * x + 3"
    action = Action(patch=correct_code, format=ActionFormat.FULL_CODE)
    obs, reward, done, truncated, info = env.step(action)

    assert done is True
    assert reward == 0.0
    assert obs.loss == 0.0
    assert obs.passed_tests == 6


def test_saddle_step_wrong_code():
    env = SaddleEnvironment(obfuscation_level=0, max_steps=10)
    obs = env.reset()
    assert obs.total_tests == 26

    action = Action(patch="class SimpleInterpreter:\n    pass", format=ActionFormat.FULL_CODE)
    obs, reward, done, truncated, info = env.step(action)

    assert done is False
    assert obs.loss > 0.0
    assert obs.passed_tests < 26
