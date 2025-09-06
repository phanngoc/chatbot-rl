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
from torch.distributions import Categorical
from openai import OpenAI


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


class PolicyNetwork(nn.Module):
    """PPO Policy Network for answer generation"""
    
    def __init__(self, hidden_size: int, memory_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.vocab_size = vocab_size
        
        # Policy head for answer generation
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size + memory_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + memory_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_state: torch.Tensor, memory_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy and value
        
        Args:
            hidden_state: Controller hidden state
            memory_context: Retrieved memory context
            
        Returns:
            logits: Policy logits for answer generation
            value: State value estimation
        """
        # Combine hidden state and memory context
        combined_input = torch.cat([hidden_state, memory_context], dim=-1)
        
        # Policy logits
        logits = self.policy_head(combined_input)
        
        # Value estimation
        value = self.value_head(combined_input)
        
        return logits, value
    
    def get_action_prob(self, logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get probability of action given logits"""
        probs = F.softmax(logits, dim=-1)
        
        # Ensure action has proper dimensions for gather
        if action.dim() == 1 and probs.dim() == 1:
            # Both are 1D, use indexing
            return probs[action]
        elif action.dim() == 1 and probs.dim() == 2:
            # action is 1D, probs is 2D, add batch dimension to action
            return probs.gather(-1, action.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        else:
            # General case with proper unsqueeze
            return probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)


class MemoryInterface(nn.Module):
    """Interface để tương tác với memory bank"""
    
    def __init__(self, hidden_size: int, memory_dim: int, vocab_size: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.vocab_size = vocab_size
        
        # Query generation
        self.query_generator = nn.Linear(hidden_size, memory_dim)
        
        # Key generation for writing
        self.key_generator = nn.Linear(hidden_size, memory_dim)
        self.value_generator = nn.Linear(hidden_size, memory_dim)
        
        # Importance weight generator
        self.importance_generator = nn.Linear(hidden_size, 1)
        
        # PPO Policy networks
        self.current_policy = PolicyNetwork(hidden_size, memory_dim, vocab_size)
        self.old_policy = PolicyNetwork(hidden_size, memory_dim, vocab_size)
        
        # Copy parameters to old policy
        self.update_old_policy()
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01
        self.value_loss_coeff = 0.5
    
    def update_old_policy(self):
        """Update old policy with current policy parameters"""
        self.old_policy.load_state_dict(self.current_policy.state_dict())
    
    def compute_ppo_importance_ratio(self, 
                                   hidden_state: torch.Tensor, 
                                   memory_context: torch.Tensor,
                                   actions: torch.Tensor) -> torch.Tensor:
        """
        Compute PPO importance ratio: ρ_θ(q, M_ret) = π_θ(y | q, M_ret) / π_old(y | q, M_ret)
        
        Args:
            hidden_state: Controller hidden state (query q)
            memory_context: Retrieved memories (M_ret)
            actions: Generated answer tokens (y)
            
        Returns:
            importance_ratio: PPO importance ratio
        """
        # Current policy probabilities
        current_logits, _ = self.current_policy(hidden_state, memory_context)
        current_probs = self.current_policy.get_action_prob(current_logits, actions)
        
        # Old policy probabilities
        with torch.no_grad():
            old_logits, _ = self.old_policy(hidden_state, memory_context)
            old_probs = self.old_policy.get_action_prob(old_logits, actions)
        
        # Importance ratio
        importance_ratio = current_probs / (old_probs + 1e-8)
        
        return importance_ratio
    
    def compute_ppo_loss(self,
                        hidden_state: torch.Tensor,
                        memory_context: torch.Tensor,
                        actions: torch.Tensor,
                        advantages: torch.Tensor,
                        returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss with clipping mechanism
        
        Args:
            hidden_state: Controller hidden state
            memory_context: Retrieved memory context
            actions: Answer tokens
            advantages: Computed advantages from answer quality
            returns: Value targets
            
        Returns:
            losses: Dictionary containing policy, value, and total loss
        """
        # Get current policy outputs
        logits, values = self.current_policy(hidden_state, memory_context)
        
        # Expand values to match batch size
        if values.dim() == 0 or (values.dim() == 1 and values.size(0) == 1):
            values = values.expand(len(returns))
        
        # Compute importance ratio
        importance_ratio = self.compute_ppo_importance_ratio(hidden_state, memory_context, actions)
        
        # PPO clipped surrogate objective
        clipped_ratio = torch.clamp(importance_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # Policy loss with clipping
        policy_loss_unclipped = importance_ratio * advantages
        policy_loss_clipped = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus for exploration
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coeff * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coeff * value_loss + entropy_loss
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'importance_ratio': importance_ratio.mean(),
            'entropy': entropy
        }
    
    def compute_advantages(self, 
                          rewards: torch.Tensor, 
                          values: torch.Tensor,
                          gamma: float = 0.99,
                          gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Rewards from answer quality evaluation
            values: Value estimates from policy network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            advantages: Computed advantages
            returns: Value targets
        """
        advantages = []
        gae = 0
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1].item() if hasattr(values[t + 1], 'item') else values[t + 1]
            
            reward_t = rewards[t].item() if hasattr(rewards[t], 'item') else rewards[t]
            value_t = values[t].item() if hasattr(values[t], 'item') else values[t]
            
            delta = reward_t + gamma * next_value - value_t
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=values.device, dtype=torch.float32)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
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
        
        # Ensure query_norm is 1D for proper dot product
        if query_norm.dim() > 1:
            query_norm = query_norm.squeeze()
            
        # If still not 1D (e.g., batch dimension), take first element
        if query_norm.dim() > 1:
            query_norm = query_norm[0]
        
        # Compute dot product similarities: keys_norm [num_memories, memory_dim] @ query_norm [memory_dim]
        similarities = torch.mv(keys_norm, query_norm)  # [num_memories]
        
        # Apply importance weights
        weighted_similarities = similarities * importance_weights
        
        # Get top-k indices
        top_k = min(top_k, len(memory_bank))
        topk_result = torch.topk(weighted_similarities, top_k)
        top_indices = topk_result.indices
        
        # Calculate attention weights
        top_similarities = weighted_similarities[top_indices]
        attention_weights = F.softmax(top_similarities, dim=0)
        
        # Weighted sum of values
        top_values = values[top_indices]  # [top_k, memory_dim]
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [top_k, 1]
        
        # Ensure proper broadcasting: [top_k, 1] * [top_k, memory_dim] = [top_k, memory_dim]
        weighted_values = attention_weights_expanded * top_values
        retrieved_memory = torch.sum(weighted_values, dim=0)  # [memory_dim]
        
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
    Memory-Augmented Neural Network (MANN) với External Working Memory
    Implement đúng theo paper: Memory Write, Memory Read, và NN Output
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
        
        # Controller network - LSTM for proper state management
        self.controller = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # External Working Memory matrices theo paper
        # Ŵ ∈ R^(N×m), V̂ ∈ R^(n×N), b̂v ∈ R^N, b̂w ∈ R^m
        self.W_hat = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)
        self.V_hat = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.b_v = nn.Parameter(torch.zeros(hidden_size))
        self.b_w = nn.Parameter(torch.zeros(output_size))
        
        # Memory interface với proper query generation
        self.memory_interface = MemoryInterface(hidden_size, memory_dim, output_size)
        
        # Memory matrix μ ∈ R^(memory_dim × memory_size)
        self.memory_matrix = nn.Parameter(torch.randn(memory_dim, memory_size) * 0.1)
        
        # Learning rate cho memory update
        self.memory_lr = 0.01
        
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
        
        # Initialize OpenAI client for embeddings (with error handling)
        try:
            self.openai_client = OpenAI()
        except Exception as e:
            self.logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
    
    def to(self, device):
        """Move model đến device cụ thể"""
        super().to(device)
        self.device = device
        
        # Move memory bank tensors to device
        for entry in self.memory_bank:
            entry.key = entry.key.to(device)
            entry.value = entry.value.to(device)
        
        # Move memory matrix to device
        self.memory_matrix = self.memory_matrix.to(device)
        
        return self
    
    def memory_write(self, z: torch.Tensor, a: torch.Tensor, q_mu: torch.Tensor, cw: float = 1.0) -> None:
        """
        Memory Write: μ̇ᵢ = -zᵢμᵢ + cwzᵢa + zᵢŴqμᵀ
        
        Args:
            z: Attention weights [memory_size]
            a: Write vector [memory_dim]
            q_mu: Controller dependent term [memory_dim]
            cw: Constant weight
        """
        # Update memory matrix using differential equation
        for i in range(min(len(z), self.memory_size)):
            z_i = z[i]
            if z_i > 0:  # Only update if attention weight > 0
                # μ̇ᵢ = -zᵢμᵢ + cwzᵢa + zᵢŴqμᵀ
                mu_i = self.memory_matrix[:, i]
                
                # Ŵqμᵀ: Ŵ shape [hidden_size, output_size], q_mu shape [memory_dim]
                # We need to project q_mu to hidden_size first
                if q_mu.size(0) != self.hidden_size:
                    # Use a simple projection or padding
                    if q_mu.size(0) < self.hidden_size:
                        q_mu_padded = torch.cat([q_mu, torch.zeros(self.hidden_size - q_mu.size(0), device=q_mu.device)])
                    else:
                        q_mu_padded = q_mu[:self.hidden_size]
                else:
                    q_mu_padded = q_mu
                
                # Ŵqμᵀ: [hidden_size, output_size] @ [hidden_size] -> [output_size]
                wq_term = self.W_hat.t() @ q_mu_padded  # [output_size]
                
                # Project back to memory_dim if needed
                if wq_term.size(0) != self.memory_dim:
                    if wq_term.size(0) < self.memory_dim:
                        wq_term = torch.cat([wq_term, torch.zeros(self.memory_dim - wq_term.size(0), device=wq_term.device)])
                    else:
                        wq_term = wq_term[:self.memory_dim]
                
                update = -z_i * mu_i + cw * z_i * a + z_i * wq_term
                
                # Update memory matrix (avoid in-place operation)
                new_mu_i = mu_i + self.memory_lr * update
                self.memory_matrix.data[:, i] = new_mu_i
        
        self.stats["total_writes"] += 1
    
    def memory_read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory Read: Mr = μz, z = softmax(μᵀq)
        
        Args:
            query: Query vector [memory_dim]
            
        Returns:
            Mr: Retrieved memory [memory_dim]
            z: Attention weights [memory_size]
        """
        # z = softmax(μᵀq)
        # μᵀ shape: [memory_size, memory_dim], query shape: [memory_dim]
        # Result: [memory_size]
        
        # Ensure query is 1D
        if query.dim() > 1:
            query = query.squeeze()
        if query.dim() > 1:
            query = query[0]
            
        z = F.softmax(torch.mv(self.memory_matrix.t(), query), dim=0)
        
        # Mr = μz
        # μ shape: [memory_dim, memory_size], z shape: [memory_size]
        # Result: [memory_dim]
        Mr = torch.mv(self.memory_matrix, z)
        
        self.stats["total_retrievals"] += 1
        return Mr, z
    
    def nn_output(self, x_tilde: torch.Tensor, Mr: torch.Tensor) -> torch.Tensor:
        """
        NN Output: uad = -Ŵᵀ(σ(V̂ᵀx̃ + b̂v) + Mr) - b̂w
        
        Args:
            x_tilde: Input vector [input_size]
            Mr: Retrieved memory [memory_dim]
            
        Returns:
            uad: Neural network output [output_size]
        """
        # σ(V̂ᵀx̃ + b̂v)
        # V̂ᵀ shape: [hidden_size, input_size], x̃ shape: [input_size]
        # Result: [hidden_size]
        hidden = torch.sigmoid(torch.mv(self.V_hat.t(), x_tilde) + self.b_v)
        
        # Add memory contribution (broadcast if needed)
        if Mr.dim() == 1 and hidden.dim() == 1:
            # Pad Mr to match hidden_size if needed
            if Mr.size(0) != hidden.size(0):
                if Mr.size(0) < hidden.size(0):
                    padding = torch.zeros(hidden.size(0) - Mr.size(0), device=Mr.device)
                    Mr_padded = torch.cat([Mr, padding])
                else:
                    Mr_padded = Mr[:hidden.size(0)]
            else:
                Mr_padded = Mr
            combined = hidden + Mr_padded
        else:
            combined = hidden + Mr
        
        # uad = -Ŵᵀ(combined) - b̂w
        # Ŵᵀ shape: [output_size, hidden_size], combined shape: [hidden_size]
        # Result: [output_size]
        uad = -torch.mv(self.W_hat.t(), combined) - self.b_w
        
        return uad
    
    def forward(self, x: torch.Tensor, retrieve_memories: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass với External Working Memory
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            retrieve_memories: Whether to retrieve from memory
            
        Returns:
            output: Model output
            memory_info: Information about retrieved memories
        """
        batch_size, seq_len, input_size = x.shape
        
        # Controller forward pass với LSTM
        controller_output, (hidden_state, cell_state) = self.controller(x)
        
        # Use last output as hidden state
        last_hidden = controller_output[:, -1, :]  # [batch_size, hidden_size]
        
        # Generate query từ hidden state
        query = self.memory_interface.generate_query(last_hidden.squeeze(0))
        
        if retrieve_memories:
            # Memory Read: Mr = μz, z = softmax(μᵀq)
            Mr, z = self.memory_read(query)
            
            # Memory Write: Generate write vector và q_mu
            write_vector = self.memory_interface.generate_key_value(last_hidden.squeeze(0))[1]
            q_mu = query  # Simplified: use query as q_mu
            
            # Update memory matrix
            self.memory_write(z, write_vector, q_mu)
            
            # NN Output: uad = -Ŵᵀ(σ(V̂ᵀx̃ + b̂v) + Mr) - b̂w
            x_tilde = x[:, -1, :].squeeze(0)  # Last input
            output = self.nn_output(x_tilde, Mr)
            
            # Prepare memory info for compatibility
            memory_info = [{
                "id": f"memory_{i}",
                "content": f"Memory slot {i}",
                "context": "External Working Memory",
                "similarity": z[i].item(),
                "attention_weight": z[i].item(),
                "importance_weight": 1.0,
                "usage_count": 0,
                "tags": ["external_memory"]
            } for i in range(min(len(z), 3))]
            
        else:
            # No memory retrieval
            output = torch.zeros(self.output_size, device=self.device)
            memory_info = []
        
        self.timestep += 1
        
        return output.unsqueeze(0), memory_info
    
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
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate OpenAI embedding for text"""
        if self.openai_client is None:
            # Fallback to simple word-based similarity if no OpenAI client
            return self._get_simple_text_vector(text)
            
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            self.logger.warning(f"Failed to get embedding: {e}")
            # Fallback to simple text vector
            return self._get_simple_text_vector(text)
    
    def _get_simple_text_vector(self, text: str) -> np.ndarray:
        """Simple fallback text vectorization"""
        # Create a simple hash-based vector for fallback
        words = text.lower().split()
        vector = np.zeros(1536)  # Same dimension as OpenAI embeddings
        
        for word in words[:100]:  # Limit to 100 words
            # Simple hash to vector index
            hash_val = hash(word) % 1536
            vector[hash_val] += 1.0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def search_memories(self, 
                       query: str,
                       top_k: int = 5,
                       min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search memories by content similarity using OpenAI embeddings
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            results: List of matching memories
        """
        if not self.memory_bank:
            return []
        
        # Generate embedding for query
        query_embedding = self._get_embedding(query)
        results = []
        
        for entry in self.memory_bank:
            # Generate embeddings for content and context
            content_embedding = self._get_embedding(entry.content)
            context_embedding = self._get_embedding(entry.context)
            
            # Calculate cosine similarity
            content_sim = self._cosine_similarity(query_embedding, content_embedding)
            context_sim = self._cosine_similarity(query_embedding, context_embedding)
            
            # Take maximum similarity
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
        
        # Update usage count for retrieved memories
        for result in results[:top_k]:
            memory_id = result["id"]
            for entry in self.memory_bank:
                if entry.id == memory_id:
                    entry.usage_count += 1
                    break
        
        # Update statistics
        self.stats["total_retrievals"] += len(results[:top_k])
        
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
        stats = {
            "total_memories": len(self.memory_bank),
            "memory_utilization": self.stats["memory_utilization"],
            "total_retrievals": self.stats["total_retrievals"],
            "total_writes": self.stats["total_writes"],
            "memory_matrix_shape": list(self.memory_matrix.shape),
            "memory_matrix_norm": torch.norm(self.memory_matrix).item(),
            "W_hat_norm": torch.norm(self.W_hat).item(),
            "V_hat_norm": torch.norm(self.V_hat).item(),
            "b_v_norm": torch.norm(self.b_v).item(),
            "b_w_norm": torch.norm(self.b_w).item()
        }
        
        if self.memory_bank:
            importance_weights = [entry.importance_weight for entry in self.memory_bank]
            usage_counts = [entry.usage_count for entry in self.memory_bank]
            
            stats.update({
                "avg_importance": np.mean(importance_weights),
                "avg_usage_count": np.mean(usage_counts),
                "max_importance": np.max(importance_weights),
                "min_importance": np.min(importance_weights)
            })
        else:
            stats.update({
                "avg_importance": 0.0,
                "avg_usage_count": 0.0,
                "max_importance": 0.0,
                "min_importance": 0.0
            })
        
        return stats
    
    def save_memory_bank(self, filepath: str) -> None:
        """Save memory bank to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "memory_bank": [entry.to_dict() for entry in self.memory_bank],
            "timestep": self.timestep,
            "stats": self.stats,
            "memory_matrix": self.memory_matrix.detach().cpu().numpy().tolist(),
            "W_hat": self.W_hat.detach().cpu().numpy().tolist(),
            "V_hat": self.V_hat.detach().cpu().numpy().tolist(),
            "b_v": self.b_v.detach().cpu().numpy().tolist(),
            "b_w": self.b_w.detach().cpu().numpy().tolist()
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
            
            # Load External Working Memory parameters
            if "memory_matrix" in data:
                self.memory_matrix.data = torch.tensor(data["memory_matrix"], device=self.device)
            if "W_hat" in data:
                self.W_hat.data = torch.tensor(data["W_hat"], device=self.device)
            if "V_hat" in data:
                self.V_hat.data = torch.tensor(data["V_hat"], device=self.device)
            if "b_v" in data:
                self.b_v.data = torch.tensor(data["b_v"], device=self.device)
            if "b_w" in data:
                self.b_w.data = torch.tensor(data["b_w"], device=self.device)
            
            self.logger.info(f"Loaded {len(self.memory_bank)} memories and External Working Memory from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load memory bank: {e}")
            return False
    
    def ppo_forward_with_memory(self, 
                               x: torch.Tensor, 
                               questions: List[str],
                               generate_answers: bool = True) -> Dict[str, Any]:
        """
        PPO forward pass with memory retrieval for answer generation
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            questions: List of questions for context
            generate_answers: Whether to generate answer probabilities
            
        Returns:
            results: Dictionary with logits, values, memory context, and retrieved memories
        """
        batch_size, seq_len, input_size = x.shape
        
        # Controller forward pass
        controller_output, (hidden_state, cell_state) = self.controller(x)
        last_hidden = controller_output[:, -1, :]  # [batch_size, hidden_size]
        
        # For PPO, we need to process each sample in the batch individually
        # For now, use the first sample for memory retrieval
        first_hidden = last_hidden[0]  # [hidden_size]
        
        # Generate query for memory retrieval
        query = self.memory_interface.generate_query(first_hidden.unsqueeze(0))  # Add batch dim back
        
        # Retrieve memories using memory interface
        retrieved_memory, memory_info = self.memory_interface.retrieve(
            query, self.memory_bank, top_k=3
        )
        
        # External Working Memory operations
        Mr, z = self.memory_read(query)
        
        # Generate write vector and update memory
        write_vector = self.memory_interface.generate_key_value(first_hidden.unsqueeze(0))[1]
        q_mu = query.squeeze(0) if query.dim() > 1 else query
        self.memory_write(z, write_vector, q_mu)
        
        # Combine retrieved memory with working memory
        combined_memory = retrieved_memory + Mr
        
        if generate_answers:
            # Get policy network outputs for answer generation
            logits, values = self.memory_interface.current_policy(
                first_hidden, 
                combined_memory
            )
        else:
            logits, values = None, None
        
        return {
            'logits': logits,
            'values': values,
            'hidden_state': first_hidden,
            'memory_context': combined_memory,
            'retrieved_memories': memory_info,
            'attention_weights': z,
            'working_memory': Mr,
            'episodic_memory': retrieved_memory
        }
    
    def compute_answer_rewards(self, 
                              generated_answers: List[str],
                              reference_answers: List[str],
                              questions: List[str]) -> torch.Tensor:
        """
        Compute rewards based on answer quality (exact match, similarity, etc.)
        
        Args:
            generated_answers: Generated answers from policy
            reference_answers: Ground truth answers
            questions: Original questions for context
            
        Returns:
            rewards: Tensor of rewards for each answer
        """
        rewards = []
        
        for gen_ans, ref_ans, question in zip(generated_answers, reference_answers, questions):
            # Simple exact match reward (can be enhanced with semantic similarity)
            if gen_ans.strip().lower() == ref_ans.strip().lower():
                reward = 1.0  # Perfect match
            else:
                # Compute word overlap similarity
                gen_words = set(gen_ans.lower().split())
                ref_words = set(ref_ans.lower().split())
                
                if len(ref_words) > 0:
                    overlap = len(gen_words.intersection(ref_words)) / len(ref_words)
                    reward = overlap
                else:
                    reward = 0.0
            
            # Bonus for using relevant memories
            memory_bonus = 0.1  # Small bonus for memory utilization
            reward += memory_bonus
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def ppo_update(self,
                   questions: List[str],
                   generated_answers: List[str],
                   reference_answers: List[str],
                   input_tensors: torch.Tensor,
                   learning_rate: float = 3e-4,
                   epochs: int = 4) -> Dict[str, float]:
        """
        PPO update step following the paper's formulation
        
        Args:
            questions: Input questions
            generated_answers: Generated answers from current policy
            reference_answers: Ground truth answers for reward computation
            input_tensors: Input tensors used for generation
            learning_rate: Learning rate for optimization
            epochs: Number of PPO epochs
            
        Returns:
            training_stats: Dictionary with training statistics
        """
        # Get current policy outputs
        forward_results = self.ppo_forward_with_memory(input_tensors, questions, generate_answers=True)
        
        hidden_state = forward_results['hidden_state']
        memory_context = forward_results['memory_context']
        current_logits = forward_results['logits']
        current_values = forward_results['values']
        
        # Convert generated answers to token indices (simplified - would need proper tokenization)
        answer_tokens = torch.randint(0, self.memory_interface.vocab_size, (len(generated_answers),))
        
        # DEBUG LOGGING START
        debug_log_path = "debug_reward_process.log"
        with open(debug_log_path, "a") as debug_file:
            debug_file.write(f"\n=== REWARD PROCESS DEBUG - {datetime.now()} ===\n")
            debug_file.write(f"Batch size: {len(generated_answers)}\n")
            debug_file.write(f"Vocab size: {self.memory_interface.vocab_size}\n")
            debug_file.write(f"Answer tokens shape: {answer_tokens.shape}\n")
            debug_file.write(f"Answer tokens sample: {answer_tokens[:5].tolist()}\n")
            
            debug_file.write("\n--- GENERATED ANSWERS ---\n")
            for i, answer in enumerate(generated_answers):
                debug_file.write(f"[{i}] Generated: {repr(answer)}\n")
            
            debug_file.write("\n--- REFERENCE ANSWERS ---\n")
            for i, ref_answer in enumerate(reference_answers):
                debug_file.write(f"[{i}] Reference: {repr(ref_answer)}\n")
            
            debug_file.write("\n--- QUESTIONS ---\n")
            for i, question in enumerate(questions):
                debug_file.write(f"[{i}] Question: {repr(question)}\n")
        
        # Compute rewards from answer quality
        rewards = self.compute_answer_rewards(generated_answers, reference_answers, questions)
        
        # DEBUG LOGGING - REWARDS
        with open(debug_log_path, "a") as debug_file:
            debug_file.write(f"\n--- REWARDS COMPUTATION ---\n")
            debug_file.write(f"Rewards shape: {rewards.shape if hasattr(rewards, 'shape') else len(rewards)}\n")
            debug_file.write(f"Rewards dtype: {rewards.dtype if hasattr(rewards, 'dtype') else type(rewards)}\n")
            debug_file.write(f"Rewards values: {rewards.tolist() if hasattr(rewards, 'tolist') else list(rewards)}\n")
            debug_file.write(f"Rewards stats - mean: {torch.mean(rewards) if hasattr(rewards, 'mean') else sum(rewards)/len(rewards):.4f}, ")
            debug_file.write(f"std: {torch.std(rewards) if hasattr(rewards, 'std') else 'N/A':.4f}, ")
            debug_file.write(f"min: {torch.min(rewards) if hasattr(rewards, 'min') else min(rewards):.4f}, ")
            debug_file.write(f"max: {torch.max(rewards) if hasattr(rewards, 'max') else max(rewards):.4f}\n")
        
        # Create values for each sample (simplified - using same value for all samples)
        batch_values = current_values.squeeze(-1).repeat(len(rewards))
        
        # DEBUG LOGGING - VALUES
        with open(debug_log_path, "a") as debug_file:
            debug_file.write(f"\n--- VALUES COMPUTATION ---\n")
            debug_file.write(f"Current values shape: {current_values.shape}\n")
            debug_file.write(f"Current values: {current_values.tolist()}\n")
            debug_file.write(f"Batch values shape: {batch_values.shape}\n")
            debug_file.write(f"Batch values: {batch_values.tolist()}\n")
        
        # Compute advantages and returns
        advantages, returns = self.memory_interface.compute_advantages(
            rewards, batch_values
        )
        
        # DEBUG LOGGING - ADVANTAGES & RETURNS
        with open(debug_log_path, "a") as debug_file:
            debug_file.write(f"\n--- ADVANTAGES & RETURNS ---\n")
            debug_file.write(f"Advantages shape: {advantages.shape}\n")
            debug_file.write(f"Advantages: {advantages.tolist()}\n")
            debug_file.write(f"Advantages stats - mean: {torch.mean(advantages):.4f}, std: {torch.std(advantages):.4f}\n")
            
            debug_file.write(f"Returns shape: {returns.shape}\n")
            debug_file.write(f"Returns: {returns.tolist()}\n")
            debug_file.write(f"Returns stats - mean: {torch.mean(returns):.4f}, std: {torch.std(returns):.4f}\n")
            debug_file.write("=== END DEBUG ===\n\n")
        
        self.logger.info(f"Debug information logged to {debug_log_path}")
        # DEBUG LOGGING END
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.memory_interface.current_policy.parameters(), lr=learning_rate)
        
        # PPO training loop
        total_losses = []
        
        for epoch in range(epochs):
            # Zero gradients first
            optimizer.zero_grad()
            
            # Compute PPO loss - detach inputs that shouldn't propagate gradients
            loss_dict = self.memory_interface.compute_ppo_loss(
                hidden_state.detach(),    # Detach hidden state 
                memory_context.detach(),  # Detach memory context
                answer_tokens,
                advantages.detach(),      # Detach advantages to avoid gradient issues
                returns.detach()          # Detach returns to avoid gradient issues
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.memory_interface.current_policy.parameters(), 0.5)
            
            optimizer.step()
            
            total_losses.append(loss_dict['total_loss'].item())
        
        # Update old policy after training
        self.memory_interface.update_old_policy()
        
        # Training statistics
        training_stats = {
            'avg_loss': np.mean(total_losses),
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'policy_entropy': loss_dict['entropy'].item(),
            'importance_ratio': loss_dict['importance_ratio'].item(),
            'memories_retrieved': len(forward_results['retrieved_memories'])
        }
        
        self.logger.info(f"PPO Update - Loss: {training_stats['avg_loss']:.4f}, "
                        f"Reward: {training_stats['avg_reward']:.4f}, "
                        f"Memories: {training_stats['memories_retrieved']}")
        
        return training_stats
    
    def generate_answer_with_ppo(self, 
                                question: str,
                                input_tensor: torch.Tensor,
                                max_length: int = 50) -> Tuple[str, List[Dict]]:
        """
        Generate answer using PPO-trained policy with memory retrieval
        
        Args:
            question: Input question
            input_tensor: Input tensor representation
            max_length: Maximum answer length
            
        Returns:
            generated_answer: Generated answer string
            memory_info: Information about retrieved memories
        """
        with torch.no_grad():
            # Forward pass with memory retrieval
            results = self.ppo_forward_with_memory(
                input_tensor.unsqueeze(0), 
                [question], 
                generate_answers=True
            )
            
            logits = results['logits']
            memory_info = results['retrieved_memories']
            
            # Sample answer tokens from policy
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
            
            # Convert tokens to answer (simplified - would need proper detokenization)
            generated_answer = f"Generated answer with token {sampled_tokens.item()}"
            
            # Add memory context to answer
            if memory_info:
                memory_context = " | ".join([mem['content'][:30] for mem in memory_info[:2]])
                generated_answer += f" (using memories: {memory_context})"
        
        return generated_answer, memory_info
