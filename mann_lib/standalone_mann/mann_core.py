"""
Core MANN Implementation
Memory-Augmented Neural Network với external memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import math
from collections import defaultdict
import json
import os
import logging
from datetime import datetime
import uuid


@dataclass
class MemoryBankEntry:
    """Entry trong memory bank"""
    id: str
    key: torch.Tensor  # Key vector để search
    value: torch.Tensor  # Value vector chứa information
    content: str  # Human-readable content
    context: str  # Context information
    usage_count: int = 0
    last_accessed: int = 0  # Timestep cuối được access
    importance_weight: float = 1.0
    timestamp: datetime = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "key": self.key.detach().cpu().numpy().tolist(),
            "value": self.value.detach().cpu().numpy().tolist(),
            "content": self.content,
            "context": self.context,
            "usage_count": self.usage_count,
            "last_accessed": self.last_accessed,
            "importance_weight": self.importance_weight,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device = None) -> 'MemoryBankEntry':
        """Create from dictionary"""
        if device is None:
            device = torch.device('cpu')
        
        return cls(
            id=data["id"],
            key=torch.tensor(data["key"], device=device),
            value=torch.tensor(data["value"], device=device),
            content=data["content"],
            context=data["context"],
            usage_count=data["usage_count"],
            last_accessed=data["last_accessed"],
            importance_weight=data["importance_weight"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tags=data["tags"],
            metadata=data["metadata"]
        )


class MemoryInterface(nn.Module):
    """Interface để tương tác với memory bank"""
    
    def __init__(self, hidden_size: int, memory_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        
        # Query generation
        self.query_generator = nn.Linear(hidden_size, memory_dim)
        
        # Key generation for writing
        self.key_generator = nn.Linear(hidden_size, memory_dim)
        self.value_generator = nn.Linear(hidden_size, memory_dim)
        
        # Importance weight generator
        self.importance_generator = nn.Linear(hidden_size, 1)
    
    def generate_query(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Generate query vector từ hidden state"""
        # Ensure hidden_state is 2D
        if hidden_state.dim() > 2:
            hidden_state = hidden_state.squeeze(0)
        return self.query_generator(hidden_state)
    
    def generate_key_value(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate key và value vectors"""
        # Ensure hidden_state is 2D
        if hidden_state.dim() > 2:
            hidden_state = hidden_state.squeeze(0)
        key = self.key_generator(hidden_state)
        value = self.value_generator(hidden_state)
        return key, value
    
    def generate_importance_weight(self, hidden_state: torch.Tensor) -> float:
        """Generate importance weight"""
        # Ensure hidden_state is 2D
        if hidden_state.dim() > 2:
            hidden_state = hidden_state.squeeze(0)
        weight = torch.sigmoid(self.importance_generator(hidden_state))
        return weight.item()
    
    def retrieve(self, 
                query_input: torch.Tensor, 
                memory_bank: List[MemoryBankEntry],
                top_k: int = 3) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Retrieve memories từ memory bank
        
        Args:
            query_input: Query vector
            memory_bank: List of memory entries
            top_k: Number of top memories to retrieve
            
        Returns:
            retrieved_memory: Weighted sum of retrieved memories
            memory_info: Information about retrieved memories
        """
        if not memory_bank:
            # Return zero vector if no memories
            zero_memory = torch.zeros(self.memory_dim, device=query_input.device)
            return zero_memory, []
        
        # Extract keys, values, and importance weights
        keys = torch.stack([entry.key for entry in memory_bank])
        values = torch.stack([entry.value for entry in memory_bank])
        importance_weights = torch.tensor([entry.importance_weight for entry in memory_bank], 
                                        device=query_input.device)
        
        # Calculate cosine similarities
        query_norm = F.normalize(query_input, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        
        # Ensure query_norm is 2D for matrix multiplication
        if query_norm.dim() == 1:
            query_norm = query_norm.unsqueeze(0)  # [1, memory_dim]
        
        similarities = torch.mm(query_norm, keys_norm.t()).squeeze(0)
        
        # Apply importance weights
        weighted_similarities = similarities * importance_weights
        
        # Get top-k indices
        top_k = min(top_k, len(memory_bank))
        top_indices = torch.topk(weighted_similarities, top_k).indices
        
        # Calculate attention weights
        top_similarities = weighted_similarities[top_indices]
        attention_weights = F.softmax(top_similarities, dim=0)
        
        # Weighted sum of values
        top_values = values[top_indices]
        retrieved_memory = torch.sum(attention_weights.unsqueeze(-1) * top_values, dim=0)
        
        # Prepare memory info
        memory_info = []
        for i, idx in enumerate(top_indices):
            entry = memory_bank[idx.item()]
            memory_info.append({
                "id": entry.id,
                "content": entry.content,
                "context": entry.context,
                "similarity": similarities[idx].item(),
                "attention_weight": attention_weights[i].item(),
                "importance_weight": entry.importance_weight,
                "usage_count": entry.usage_count,
                "tags": entry.tags
            })
        
        return retrieved_memory, memory_info


class MemoryAugmentedNetwork(nn.Module):
    """
    Memory-Augmented Neural Network (MANN)
    Sử dụng external memory để store và retrieve experiences
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 memory_size: int,
                 memory_dim: int,
                 output_size: int,
                 device: torch.device = None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.output_size = output_size
        self.device = device or torch.device('cpu')
        
        # Controller network - use simple linear layers instead of LSTM for now
        self.controller = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Memory interface
        self.memory_interface = MemoryInterface(hidden_size, memory_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size + memory_dim, output_size)
        
        # Initialize memory bank
        self.memory_bank: List[MemoryBankEntry] = []
        self.timestep = 0
        
        # Statistics
        self.stats = {
            "total_retrievals": 0,
            "total_writes": 0,
            "memory_utilization": 0.0,
            "avg_retrieval_accuracy": 0.0
        }
        
        self.logger = logging.getLogger("MANN")
    
    def to(self, device):
        """Move model đến device cụ thể"""
        super().to(device)
        self.device = device
        
        # Move memory bank tensors to device
        for entry in self.memory_bank:
            entry.key = entry.key.to(device)
            entry.value = entry.value.to(device)
        
        return self
    
    def forward(self, x: torch.Tensor, retrieve_memories: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            retrieve_memories: Whether to retrieve from memory
            
        Returns:
            output: Model output
            memory_info: Information about retrieved memories
        """
        # Controller forward pass
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Reshape to [batch_size * seq_len, input_size] for linear layers
        x_reshaped = x.view(-1, input_size)
        
        # Process through controller
        controller_output = self.controller(x_reshaped)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        controller_output = controller_output.view(batch_size, seq_len, -1)
        
        # Use last output as hidden state
        last_hidden = controller_output[:, -1, :]  # [batch_size, hidden_size]
        
        if retrieve_memories and self.memory_bank:
            # Generate query
            query = self.memory_interface.generate_query(last_hidden)
            
            # Retrieve memories
            retrieved_memory, memory_info = self.memory_interface.retrieve(
                query, self.memory_bank, top_k=3
            )
            
            # Update statistics
            self.stats["total_retrievals"] += 1
            
            # Update access counts
            for info in memory_info:
                memory_id = info["id"]
                for entry in self.memory_bank:
                    if entry.id == memory_id:
                        entry.usage_count += 1
                        entry.last_accessed = self.timestep
                        break
        else:
            retrieved_memory = torch.zeros(self.memory_dim, device=self.device)
            memory_info = []
        
        # Combine controller output with retrieved memory
        if retrieved_memory.dim() == 1:
            retrieved_memory = retrieved_memory.unsqueeze(0)
        combined_input = torch.cat([last_hidden, retrieved_memory], dim=-1)
        
        # Final output
        output = self.output_layer(combined_input)
        
        self.timestep += 1
        
        return output, memory_info
    
    def add_memory(self, 
                   content: str,
                   context: str = "",
                   tags: List[str] = None,
                   importance_weight: float = 1.0,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add new memory to memory bank
        
        Args:
            content: Memory content
            context: Context information
            tags: Memory tags
            importance_weight: Importance weight
            metadata: Additional metadata
            
        Returns:
            memory_id: ID of added memory
        """
        # Generate dummy hidden state for key-value generation
        dummy_hidden = torch.zeros(1, self.hidden_size, device=self.device)
        
        # Generate key and value
        key, value = self.memory_interface.generate_key_value(dummy_hidden)
        key = key.squeeze(0)
        value = value.squeeze(0)
        
        # Generate importance weight
        importance = self.memory_interface.generate_importance_weight(dummy_hidden)
        importance = max(importance, importance_weight)  # Use provided weight if higher
        
        # Create memory entry
        memory_id = str(uuid.uuid4())
        entry = MemoryBankEntry(
            id=memory_id,
            key=key,
            value=value,
            content=content,
            context=context,
            importance_weight=importance,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Add to memory bank
        if len(self.memory_bank) >= self.memory_size:
            # Remove least important memory
            self._remove_least_important_memory()
        
        self.memory_bank.append(entry)
        self.stats["total_writes"] += 1
        self.stats["memory_utilization"] = len(self.memory_bank) / self.memory_size
        
        self.logger.info(f"Added memory {memory_id}: {content[:50]}...")
        
        return memory_id
    
    def update_memory(self, 
                     memory_id: str,
                     content: str = None,
                     context: str = None,
                     importance_weight: float = None,
                     tags: List[str] = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Update existing memory
        
        Args:
            memory_id: ID of memory to update
            content: New content
            context: New context
            importance_weight: New importance weight
            tags: New tags
            metadata: New metadata
            
        Returns:
            success: Whether update was successful
        """
        for entry in self.memory_bank:
            if entry.id == memory_id:
                if content is not None:
                    entry.content = content
                if context is not None:
                    entry.context = context
                if importance_weight is not None:
                    entry.importance_weight = importance_weight
                if tags is not None:
                    entry.tags = tags
                if metadata is not None:
                    entry.metadata.update(metadata)
                
                # Update key and value if content changed
                if content is not None:
                    dummy_hidden = torch.zeros(1, self.hidden_size, device=self.device)
                    key, value = self.memory_interface.generate_key_value(dummy_hidden)
                    entry.key = key.squeeze(0)
                    entry.value = value.squeeze(0)
                
                self.logger.info(f"Updated memory {memory_id}")
                return True
        
        return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete memory from memory bank
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            success: Whether deletion was successful
        """
        for i, entry in enumerate(self.memory_bank):
            if entry.id == memory_id:
                del self.memory_bank[i]
                self.stats["memory_utilization"] = len(self.memory_bank) / self.memory_size
                self.logger.info(f"Deleted memory {memory_id}")
                return True
        
        return False
    
    def search_memories(self, 
                       query: str,
                       top_k: int = 5,
                       min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search memories by content similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            results: List of matching memories
        """
        if not self.memory_bank:
            return []
        
        # Simple text-based similarity (in production, use embeddings)
        query_lower = query.lower()
        results = []
        
        for entry in self.memory_bank:
            content_lower = entry.content.lower()
            context_lower = entry.context.lower()
            
            # Calculate simple similarity
            content_words = set(content_lower.split())
            context_words = set(context_lower.split())
            query_words = set(query_lower.split())
            
            content_sim = len(content_words.intersection(query_words)) / max(len(query_words), 1)
            context_sim = len(context_words.intersection(query_words)) / max(len(query_words), 1)
            
            similarity = max(content_sim, context_sim)
            
            if similarity >= min_similarity:
                results.append({
                    "id": entry.id,
                    "content": entry.content,
                    "context": entry.context,
                    "similarity": similarity,
                    "importance_weight": entry.importance_weight,
                    "usage_count": entry.usage_count,
                    "tags": entry.tags,
                    "timestamp": entry.timestamp.isoformat()
                })
        
        # Sort by similarity and importance
        results.sort(key=lambda x: x["similarity"] * x["importance_weight"], reverse=True)
        
        return results[:top_k]
    
    def _remove_least_important_memory(self) -> None:
        """Remove least important memory from bank"""
        if not self.memory_bank:
            return
        
        # Find least important memory
        least_important_idx = 0
        min_score = float('inf')
        
        for i, entry in enumerate(self.memory_bank):
            # Score based on importance and usage
            score = entry.importance_weight / (1 + entry.usage_count)
            if score < min_score:
                min_score = score
                least_important_idx = i
        
        removed_entry = self.memory_bank.pop(least_important_idx)
        self.logger.info(f"Removed least important memory: {removed_entry.id}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        if not self.memory_bank:
            return {"total_memories": 0}
        
        importance_weights = [entry.importance_weight for entry in self.memory_bank]
        usage_counts = [entry.usage_count for entry in self.memory_bank]
        
        return {
            "total_memories": len(self.memory_bank),
            "memory_utilization": self.stats["memory_utilization"],
            "total_retrievals": self.stats["total_retrievals"],
            "total_writes": self.stats["total_writes"],
            "avg_importance": np.mean(importance_weights),
            "avg_usage_count": np.mean(usage_counts),
            "max_importance": np.max(importance_weights),
            "min_importance": np.min(importance_weights)
        }
    
    def save_memory_bank(self, filepath: str) -> None:
        """Save memory bank to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "memory_bank": [entry.to_dict() for entry in self.memory_bank],
            "timestep": self.timestep,
            "stats": self.stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_memory_bank(self, filepath: str) -> bool:
        """Load memory bank from file"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load memory bank
            self.memory_bank = [
                MemoryBankEntry.from_dict(entry_data, self.device)
                for entry_data in data["memory_bank"]
            ]
            
            # Load timestep and stats
            self.timestep = data.get("timestep", 0)
            self.stats.update(data.get("stats", {}))
            
            self.logger.info(f"Loaded {len(self.memory_bank)} memories from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load memory bank: {e}")
            return False
