# Backward compatibility: re-export from new locations
# Old code can still do `from agent import LLMKernel, SemanticMomentumOptimizer`
from agents.llm_kernel import LLMKernel
from agents.optimizer import CodeOptimizer, AdaptiveOptimizer
from agents.memory_buffer import MemoryBuffer
from agents.momentum_agent import SemanticMomentumOptimizer, BaselineAgent

__all__ = [
    'LLMKernel', 'CodeOptimizer', 'AdaptiveOptimizer', 'MemoryBuffer',
    'SemanticMomentumOptimizer', 'BaselineAgent',
]
