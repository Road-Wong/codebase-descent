# Backward compatibility: re-export from new locations
# Old code can still do `from benchmarks import ConvexEnvironment`
from environments.env_convex import ConvexEnvironment
from environments.env_saddle import SaddleEnvironment
from environments.env_lru_cache import LRUCacheEnvironment
from environments.env_graph import GraphEnvironment
from environments.env_order_discount import OrderDiscountEnvironment
from environments.obfuscator import obfuscate_code, create_obfuscode_benchmark, SemanticStripper
from core.env_base import AbstractEnvironment

__all__ = [
    'AbstractEnvironment', 'ConvexEnvironment', 'SaddleEnvironment',
    'LRUCacheEnvironment', 'GraphEnvironment', 'OrderDiscountEnvironment',
    'obfuscate_code', 'create_obfuscode_benchmark', 'SemanticStripper',
]
