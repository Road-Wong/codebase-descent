"""
Tests for full pipeline: SGD evaluator + CodeEnv + SEARCH/REPLACE (from test_sgd_smo.py).
"""

from core.env_base import CodeEnv
from core.types import Action, ActionFormat
from core.protocol import make_action


def test_full_pipeline_broken_to_fixed():
    env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=20)
    obs = env.reset()

    # Break the code
    broken = env.ground_truth.replace("left = left + right", "left = left - right", 1)
    obs = env.reset(initial_code=broken)
    assert obs.loss > 0

    # Fix with SEARCH/REPLACE
    patch = make_action("left = left - right", "left = left + right")
    action = Action(patch=patch, format=ActionFormat.SEARCH_REPLACE)
    obs2, reward, done, truncated, info = env.step(action)

    assert done is True
    assert obs2.loss == 0.0
