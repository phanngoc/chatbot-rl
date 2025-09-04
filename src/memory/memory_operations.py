"""
Memory Operations Enums và Constants
Được define riêng để tránh circular imports
"""

from enum import Enum

class MemoryOperation(Enum):
    """Các operations cho memory management"""
    ADD = "ADD"
    UPDATE = "UPDATE" 
    DELETE = "DELETE"
    NOOP = "NOOP"
