"""
Tests for SMO structural properties (from test_sgd_smo.py).
"""

from agents.momentum_agent import SemanticMomentumOptimizer
from agents.base_agent import DiffAgent


def test_smo_inherits_diff_agent():
    assert issubclass(SemanticMomentumOptimizer, DiffAgent)


def test_smo_has_required_methods():
    assert hasattr(SemanticMomentumOptimizer, "_extract_gradient")
    assert hasattr(SemanticMomentumOptimizer, "_update_momentum")
    assert hasattr(SemanticMomentumOptimizer, "_generate_with_momentum")
