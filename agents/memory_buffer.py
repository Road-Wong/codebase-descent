from typing import List, Dict, Any
from collections import deque


class MemoryBuffer:
    """Ring buffer for storing optimization history."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, item: Dict[str, Any]):
        self.buffer.append(item)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        return list(self.buffer)[-n:] if n <= len(self.buffer) else list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
