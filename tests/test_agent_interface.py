"""
Tests for agent interface (from test_new_architecture.py).
"""

from agents.momentum_agent import SemanticMomentumOptimizer, BaselineAgent


def test_smo_has_required_methods():
    assert hasattr(SemanticMomentumOptimizer, "act")
    assert hasattr(SemanticMomentumOptimizer, "reset")


def test_baseline_has_required_methods():
    assert hasattr(BaselineAgent, "act")
    assert hasattr(BaselineAgent, "reset")
