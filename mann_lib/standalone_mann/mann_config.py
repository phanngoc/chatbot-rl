"""
MANN Configuration Management
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path


@dataclass
class MANNConfig:
    """Configuration cho MANN system"""
    
    # Model parameters
    input_size: int = 768
    hidden_size: int = 256
    memory_size: int = 1000
    memory_dim: int = 128
    output_size: int = 768
    
    # Memory management
    similarity_threshold_update: float = 0.8
    similarity_threshold_delete: float = 0.95
    importance_threshold: float = 0.3
    max_memory_capacity: int = 5000
    
    # Learning parameters
    meta_learning_rate: float = 1e-4
    adaptation_steps: int = 3
    batch_size: int = 32
    
    # API settings
    api_host: str = "localhost"
    api_port: int = 8000
    api_timeout: int = 30
    
    # Storage settings
    data_dir: str = "./data"
    model_save_path: str = "./models/mann_model.pt"
    memory_save_path: str = "./data/memory_bank.pkl"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/mann.log"
    
    # Production settings
    enable_monitoring: bool = True
    enable_pager: bool = True
    pager_webhook_url: Optional[str] = None
    health_check_interval: int = 60  # seconds
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    def __post_init__(self):
        """Initialize paths and directories"""
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "models").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        (self.data_dir / "cache").mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "memory_size": self.memory_size,
            "memory_dim": self.memory_dim,
            "output_size": self.output_size,
            "similarity_threshold_update": self.similarity_threshold_update,
            "similarity_threshold_delete": self.similarity_threshold_delete,
            "importance_threshold": self.importance_threshold,
            "max_memory_capacity": self.max_memory_capacity,
            "meta_learning_rate": self.meta_learning_rate,
            "adaptation_steps": self.adaptation_steps,
            "batch_size": self.batch_size,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_timeout": self.api_timeout,
            "data_dir": str(self.data_dir),
            "model_save_path": self.model_save_path,
            "memory_save_path": self.memory_save_path,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_monitoring": self.enable_monitoring,
            "enable_pager": self.enable_pager,
            "pager_webhook_url": self.pager_webhook_url,
            "health_check_interval": self.health_check_interval,
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MANNConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'MANNConfig':
        """Load config from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str) -> None:
        """Save config to JSON file"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update_from_env(self) -> None:
        """Update config from environment variables"""
        env_mapping = {
            "MANN_INPUT_SIZE": "input_size",
            "MANN_HIDDEN_SIZE": "hidden_size", 
            "MANN_MEMORY_SIZE": "memory_size",
            "MANN_MEMORY_DIM": "memory_dim",
            "MANN_OUTPUT_SIZE": "output_size",
            "MANN_API_HOST": "api_host",
            "MANN_API_PORT": "api_port",
            "MANN_LOG_LEVEL": "log_level",
            "MANN_ENABLE_PAGER": "enable_pager",
            "MANN_PAGER_WEBHOOK": "pager_webhook_url"
        }
        
        for env_var, attr_name in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if attr_name in ["input_size", "hidden_size", "memory_size", "memory_dim", "output_size", "api_port", "max_memory_capacity", "adaptation_steps", "batch_size", "api_timeout", "health_check_interval", "cache_size", "cache_ttl"]:
                    value = int(value)
                elif attr_name in ["similarity_threshold_update", "similarity_threshold_delete", "importance_threshold", "meta_learning_rate"]:
                    value = float(value)
                elif attr_name == "enable_pager":
                    value = value.lower() in ("true", "1", "yes", "on")
                
                setattr(self, attr_name, value)
