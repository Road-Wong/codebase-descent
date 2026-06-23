"""
Tests for backward compatibility with old import paths (from test_new_architecture.py).
"""

from benchmarks import ConvexEnvironment, SaddleEnvironment
from benchmarks import obfuscate_code, SemanticStripper
from agent import LLMKernel, SemanticMomentumOptimizer, BaselineAgent
from harness import DynamicsAuditor, NoiseInjector


def test_benchmarks_reexport():
    from environments import ConvexEnvironment as NewConvex
    assert ConvexEnvironment is NewConvex


def test_agent_reexport():
    from agents import LLMKernel as NewLLM
    assert LLMKernel is NewLLM
