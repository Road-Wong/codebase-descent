from typing import List, Dict, Any
from collections import deque


class MemoryBuffer:
    """
    Ring buffer for storing optimization history.
    Simulates context window and momentum in optimization.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize memory buffer.
        
        Args:
            max_size: Maximum number of items to store (context window size)
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, item: Dict[str, Any]):
        """
        Add item to buffer.
        
        Args:
            item: Dictionary containing code, feedback, etc.
        """
        self.buffer.append(item)
        
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get all items in buffer.
        
        Returns:
            List of historical items
        """
        return list(self.buffer)
    
    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """
        Get n most recent items.
        
        Args:
            n: Number of items to retrieve
            
        Returns:
            List of recent items
        """
        return list(self.buffer)[-n:] if n <= len(self.buffer) else list(self.buffer)
    
    def clear(self):
        """Clear all items from buffer."""
        self.buffer.clear()
    
    def __len__(self):
        """Return number of items in buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.max_size
