"""
Meta-learning với Episodic Memory
Học cách chọn lọc và kích hoạt trải nghiệm phù hợp
Implement MANN, NTM, DNC patterns
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


@dataclass
class MemoryBankEntry:
    """Entry trong memory bank"""
    key: torch.Tensor  # Key vector để search
    value: torch.Tensor  # Value vector chứa information
    usage_count: int = 0
    last_accessed: int = 0  # Timestep cuối được access
    importance_weight: float = 1.0


class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine implementation
    Có external memory và attention mechanism
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 memory_size: int,
                 memory_dim: int,
                 num_heads: int = 1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Controller (LSTM)
        self.controller = nn.LSTM(input_size + memory_dim * num_heads, hidden_size)
        
        # Memory interaction heads
        self.read_heads = nn.ModuleList([
            ReadHead(hidden_size, memory_dim) for _ in range(num_heads)
        ])
        self.write_head = WriteHead(hidden_size, memory_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size + memory_dim * num_heads, input_size)
        
        # Initialize memory
        self.register_buffer('memory', torch.randn(memory_size, memory_dim) * 0.01)
        self.register_buffer('read_weights', torch.zeros(num_heads, memory_size))
        self.register_buffer('write_weights', torch.zeros(memory_size))
        
    def forward(self, inputs: torch.Tensor, hidden_state=None):
        """
        Forward pass qua NTM
        
        Args:
            inputs: (seq_len, batch_size, input_size)
            hidden_state: Hidden state của controller
        """
        batch_size = inputs.size(1)
        seq_len = inputs.size(0)
        
        outputs = []
        
        # Initialize hidden state if None
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, batch_size, self.hidden_size, device=inputs.device),
                torch.zeros(1, batch_size, self.hidden_size, device=inputs.device)
            )
        
        for t in range(seq_len):
            # Read from memory
            read_vectors = []
            for head_idx, read_head in enumerate(self.read_heads):
                read_vector = read_head(self.memory, self.read_weights[head_idx])
                read_vectors.append(read_vector)
            
            read_vector = torch.cat(read_vectors, dim=-1)  # (batch_size, memory_dim * num_heads)
            
            # Controller input: current input + read vectors
            controller_input = torch.cat([inputs[t], read_vector], dim=-1)
            controller_input = controller_input.unsqueeze(0)  # (1, batch_size, input_size + memory_dim)
            
            # Controller forward
            controller_output, hidden_state = self.controller(controller_input, hidden_state)
            controller_output = controller_output.squeeze(0)  # (batch_size, hidden_size)
            
            # Write to memory
            self.memory, self.write_weights = self.write_head(
                controller_output, self.memory, self.write_weights
            )
            
            # Update read weights
            for head_idx, read_head in enumerate(self.read_heads):
                self.read_weights[head_idx] = read_head.get_attention_weights(
                    controller_output, self.memory, self.read_weights[head_idx]
                )
            
            # Generate output
            output_input = torch.cat([controller_output, read_vector], dim=-1)
            output = self.output_layer(output_input)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0), hidden_state


class ReadHead(nn.Module):
    """Read head cho NTM"""
    
    def __init__(self, controller_size: int, memory_dim: int):
        super().__init__()
        
        self.controller_size = controller_size
        self.memory_dim = memory_dim
        
        # Parameters cho addressing
        self.key_layer = nn.Linear(controller_size, memory_dim)
        self.key_strength_layer = nn.Linear(controller_size, 1)
        self.interpolation_gate_layer = nn.Linear(controller_size, 1)
        self.shift_layer = nn.Linear(controller_size, 3)  # shift weights
        self.sharpening_layer = nn.Linear(controller_size, 1)
    
    def forward(self, memory: torch.Tensor, prev_weights: torch.Tensor) -> torch.Tensor:
        """Read từ memory dựa trên attention weights"""
        # memory: (memory_size, memory_dim)
        # prev_weights: (memory_size,)
        
        read_vector = torch.matmul(prev_weights.unsqueeze(0), memory)  # (1, memory_dim)
        return read_vector.squeeze(0)  # (memory_dim,)
    
    def get_attention_weights(self, 
                            controller_output: torch.Tensor,
                            memory: torch.Tensor,
                            prev_weights: torch.Tensor) -> torch.Tensor:
        """Tính attention weights cho reading"""
        batch_size = controller_output.size(0)
        
        # Content-based addressing
        key = self.key_layer(controller_output)  # (batch_size, memory_dim)
        key_strength = F.softplus(self.key_strength_layer(controller_output))  # (batch_size, 1)
        
        # Cosine similarity
        key_norm = F.normalize(key, p=2, dim=-1)
        memory_norm = F.normalize(memory, p=2, dim=-1)
        cosine_sim = torch.matmul(key_norm, memory_norm.t())  # (batch_size, memory_size)
        
        content_weights = F.softmax(key_strength * cosine_sim, dim=-1)  # (batch_size, memory_size)
        
        # Interpolation với previous weights
        interpolation_gate = torch.sigmoid(self.interpolation_gate_layer(controller_output))  # (batch_size, 1)
        gated_weights = interpolation_gate * content_weights + (1 - interpolation_gate) * prev_weights.unsqueeze(0)
        
        # Convolutional shift
        shift_weights = F.softmax(self.shift_layer(controller_output), dim=-1)  # (batch_size, 3)
        shifted_weights = self._convolutional_shift(gated_weights, shift_weights)
        
        # Sharpening
        sharpening = F.softplus(self.sharpening_layer(controller_output)) + 1  # (batch_size, 1)
        final_weights = torch.pow(shifted_weights, sharpening)
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return final_weights.squeeze(0)  # (memory_size,)
    
    def _convolutional_shift(self, weights: torch.Tensor, shift_weights: torch.Tensor) -> torch.Tensor:
        """Convolutional shift operation"""
        batch_size, memory_size = weights.size()
        
        # Pad weights for circular convolution
        padded_weights = torch.cat([weights[:, -1:], weights, weights[:, :1]], dim=-1)
        
        # Apply shift
        shifted = torch.zeros_like(weights)
        for i in range(3):
            shift_amount = i - 1  # -1, 0, 1
            if shift_amount == 0:
                shifted += shift_weights[:, i:i+1] * weights
            else:
                shifted += shift_weights[:, i:i+1] * torch.roll(weights, shift_amount, dims=-1)
        
        return shifted


class WriteHead(nn.Module):
    """Write head cho NTM"""
    
    def __init__(self, controller_size: int, memory_dim: int):
        super().__init__()
        
        self.controller_size = controller_size
        self.memory_dim = memory_dim
        
        # Write parameters
        self.key_layer = nn.Linear(controller_size, memory_dim)
        self.key_strength_layer = nn.Linear(controller_size, 1)
        self.interpolation_gate_layer = nn.Linear(controller_size, 1)
        self.shift_layer = nn.Linear(controller_size, 3)
        self.sharpening_layer = nn.Linear(controller_size, 1)
        
        # Write operations
        self.erase_layer = nn.Linear(controller_size, memory_dim)
        self.add_layer = nn.Linear(controller_size, memory_dim)
    
    def forward(self, 
                controller_output: torch.Tensor,
                memory: torch.Tensor,
                prev_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write vào memory"""
        batch_size = controller_output.size(0)
        
        # Get write weights (similar to read head)
        write_weights = self._get_write_weights(controller_output, memory, prev_weights)
        
        # Erase and add operations
        erase_vector = torch.sigmoid(self.erase_layer(controller_output))  # (batch_size, memory_dim)
        add_vector = self.add_layer(controller_output)  # (batch_size, memory_dim)
        
        # Update memory
        # Erase: M_t = M_{t-1} * (1 - w_t * e_t)
        erase_term = write_weights.unsqueeze(-1) * erase_vector.unsqueeze(1)  # (batch_size, memory_size, memory_dim)
        erased_memory = memory.unsqueeze(0) * (1 - erase_term)  # (batch_size, memory_size, memory_dim)
        
        # Add: M_t = erased_memory + w_t * a_t
        add_term = write_weights.unsqueeze(-1) * add_vector.unsqueeze(1)  # (batch_size, memory_size, memory_dim)
        updated_memory = erased_memory + add_term
        
        # Average across batch (assuming batch_size=1 for simplicity)
        updated_memory = updated_memory.mean(dim=0)  # (memory_size, memory_dim)
        write_weights = write_weights.mean(dim=0)  # (memory_size,)
        
        return updated_memory, write_weights
    
    def _get_write_weights(self, 
                          controller_output: torch.Tensor,
                          memory: torch.Tensor,
                          prev_weights: torch.Tensor) -> torch.Tensor:
        """Tính write weights (tương tự read head)"""
        # Reuse logic từ ReadHead
        read_head = ReadHead(self.controller_size, self.memory_dim)
        read_head.key_layer = self.key_layer
        read_head.key_strength_layer = self.key_strength_layer
        read_head.interpolation_gate_layer = self.interpolation_gate_layer
        read_head.shift_layer = self.shift_layer
        read_head.sharpening_layer = self.sharpening_layer
        
        return read_head.get_attention_weights(controller_output, memory, prev_weights)


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
                 output_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.output_size = output_size
        
        # Controller network
        self.controller = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Memory interface
        self.memory_interface = MemoryInterface(hidden_size, memory_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size + memory_dim, output_size)
        
        # Initialize memory bank
        self.memory_bank = [
            MemoryBankEntry(
                key=torch.randn(memory_dim),
                value=torch.randn(memory_dim)
            ) for _ in range(memory_size)
        ]
        
        self.timestep = 0
    
    def forward(self, x: torch.Tensor, retrieve_memories: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            retrieve_memories: Whether to retrieve from memory
        
        Returns:
            output: Model output
            retrieved_memories: List of retrieved memory information
        """
        batch_size, seq_len, _ = x.size()
        
        # Controller forward
        controller_output, _ = self.controller(x)  # (batch_size, seq_len, hidden_size)
        
        outputs = []
        retrieved_memories_list = []
        
        for t in range(seq_len):
            timestep_output = controller_output[:, t, :]  # (batch_size, hidden_size)
            
            if retrieve_memories:
                # Retrieve relevant memories
                memory_output, retrieved_info = self.memory_interface.retrieve(
                    timestep_output, self.memory_bank
                )
                retrieved_memories_list.append(retrieved_info)
            else:
                memory_output = torch.zeros(batch_size, self.memory_dim, device=x.device)
                retrieved_memories_list.append([])
            
            # Combine controller output with memory
            combined_output = torch.cat([timestep_output, memory_output], dim=-1)
            output = self.output_layer(combined_output)
            outputs.append(output)
            
            self.timestep += 1
        
        final_output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        return final_output, retrieved_memories_list
    
    def store_experience(self, 
                        context: torch.Tensor, 
                        response: torch.Tensor,
                        reward: float) -> None:
        """Store experience vào memory bank"""
        # Generate key from context
        with torch.no_grad():
            controller_output, _ = self.controller(context.unsqueeze(0))
            key = controller_output.squeeze(0).mean(dim=0)  # Average over sequence
        
        # Generate value from response
        with torch.no_grad():
            response_output, _ = self.controller(response.unsqueeze(0))
            value = response_output.squeeze(0).mean(dim=0)
        
        # Find least recently used memory slot
        lru_idx = min(range(len(self.memory_bank)), 
                     key=lambda i: self.memory_bank[i].last_accessed)
        
        # Update memory entry
        importance_weight = max(0.1, min(2.0, 1.0 + reward))  # Convert reward to importance
        
        self.memory_bank[lru_idx] = MemoryBankEntry(
            key=key.detach(),
            value=value.detach(),
            usage_count=0,
            last_accessed=self.timestep,
            importance_weight=importance_weight
        )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Thống kê về memory bank"""
        usage_counts = [entry.usage_count for entry in self.memory_bank]
        importance_weights = [entry.importance_weight for entry in self.memory_bank]
        
        return {
            "total_memories": len(self.memory_bank),
            "avg_usage_count": np.mean(usage_counts),
            "max_usage_count": max(usage_counts),
            "avg_importance": np.mean(importance_weights),
            "high_importance_memories": sum(1 for w in importance_weights if w > 1.5),
            "current_timestep": self.timestep
        }


class MemoryInterface(nn.Module):
    """Interface để interact với memory bank"""
    
    def __init__(self, controller_size: int, memory_dim: int):
        super().__init__()
        
        self.controller_size = controller_size
        self.memory_dim = memory_dim
        
        # Query generation
        self.query_layer = nn.Linear(controller_size, memory_dim)
        
        # Attention mechanism
        self.attention_layer = nn.Linear(memory_dim, 1)
        
        # Memory combination
        self.combine_layer = nn.Linear(memory_dim, memory_dim)
    
    def retrieve(self, 
                query_input: torch.Tensor, 
                memory_bank: List[MemoryBankEntry],
                top_k: int = 3) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Retrieve memories based on query
        
        Args:
            query_input: Controller output để tạo query
            memory_bank: Memory bank để search
            top_k: Number of memories to retrieve
        
        Returns:
            memory_output: Combined memory output
            retrieved_info: Information về retrieved memories
        """
        batch_size = query_input.size(0)
        
        # Generate query
        query = self.query_layer(query_input)  # (batch_size, memory_dim)
        
        # Calculate similarities với all memories
        similarities = []
        memory_values = []
        memory_info = []
        
        for i, entry in enumerate(memory_bank):
            # Cosine similarity
            key_norm = F.normalize(entry.key.unsqueeze(0), p=2, dim=-1)
            query_norm = F.normalize(query, p=2, dim=-1)
            similarity = torch.matmul(query_norm, key_norm.t()).squeeze()  # (batch_size,)
            
            # Weight by importance và usage
            weighted_similarity = similarity * entry.importance_weight
            
            similarities.append(weighted_similarity)
            memory_values.append(entry.value.unsqueeze(0).expand(batch_size, -1))
            memory_info.append({
                "index": i,
                "similarity": similarity.item() if batch_size == 1 else similarity.mean().item(),
                "importance_weight": entry.importance_weight,
                "usage_count": entry.usage_count
            })
        
        # Get top-k memories
        similarities_tensor = torch.stack(similarities, dim=-1)  # (batch_size, memory_size)
        top_k_indices = torch.topk(similarities_tensor, min(top_k, len(memory_bank)), dim=-1).indices
        
        # Retrieve top-k memories
        retrieved_memories = []
        retrieved_info = []
        
        for batch_idx in range(batch_size):
            batch_memories = []
            batch_info = []
            
            for k_idx in range(min(top_k, len(memory_bank))):
                memory_idx = top_k_indices[batch_idx, k_idx].item()
                batch_memories.append(memory_values[memory_idx][batch_idx])
                batch_info.append(memory_info[memory_idx])
                
                # Update usage count
                memory_bank[memory_idx].usage_count += 1
                memory_bank[memory_idx].last_accessed = getattr(self, 'timestep', 0)
            
            retrieved_memories.append(torch.stack(batch_memories))  # (top_k, memory_dim)
            retrieved_info.append(batch_info)
        
        # Combine retrieved memories
        retrieved_tensor = torch.stack(retrieved_memories)  # (batch_size, top_k, memory_dim)
        
        # Attention over retrieved memories
        attention_scores = F.softmax(
            self.attention_layer(retrieved_tensor).squeeze(-1), dim=-1
        )  # (batch_size, top_k)
        
        # Weighted combination
        memory_output = torch.sum(
            attention_scores.unsqueeze(-1) * retrieved_tensor, dim=1
        )  # (batch_size, memory_dim)
        
        # Transform output
        memory_output = self.combine_layer(memory_output)
        
        return memory_output, retrieved_info


class MetaLearningEpisodicSystem:
    """
    Main system cho meta-learning với episodic memory
    Học cách select và activate relevant experiences
    """
    
    def __init__(self,
                 input_size: int = 768,
                 hidden_size: int = 256,
                 memory_size: int = 1000,
                 memory_dim: int = 128,
                 output_size: int = 768):
        
        # Initialize MANN
        self.mann = MemoryAugmentedNetwork(
            input_size, hidden_size, memory_size, memory_dim, output_size
        )
        
        # Meta-learning components
        self.meta_optimizer = torch.optim.Adam(self.mann.parameters(), lr=1e-4)
        self.experience_buffer = []
        self.adaptation_steps = 3
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.meta_learning_stats = {
            "episodes_trained": 0,
            "avg_adaptation_loss": 0.0,
            "memory_utilization": 0.0
        }
    
    def meta_train_episode(self, 
                          support_data: List[Dict],
                          query_data: List[Dict]) -> Dict[str, float]:
        """
        Meta-training episode với support và query sets
        
        Args:
            support_data: Support set để adapt
            query_data: Query set để evaluate
        """
        # Save original parameters
        original_params = {name: param.clone() for name, param in self.mann.named_parameters()}
        
        # Inner loop: Adapt trên support set
        adaptation_losses = []
        for step in range(self.adaptation_steps):
            support_loss = self._compute_support_loss(support_data)
            
            # Gradient step
            self.meta_optimizer.zero_grad()
            support_loss.backward(retain_graph=True)
            
            # Manual parameter update (simplified MAML)
            with torch.no_grad():
                for name, param in self.mann.named_parameters():
                    if param.grad is not None:
                        param.data = param.data - 0.01 * param.grad.data
            
            adaptation_losses.append(support_loss.item())
        
        # Outer loop: Evaluate trên query set
        query_loss = self._compute_query_loss(query_data)
        
        # Meta-gradient step
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        
        # Restore original parameters cho next episode
        with torch.no_grad():
            for name, param in self.mann.named_parameters():
                param.data = original_params[name].data
        
        # Update stats
        self.meta_learning_stats["episodes_trained"] += 1
        self.meta_learning_stats["avg_adaptation_loss"] = np.mean(adaptation_losses)
        
        return {
            "adaptation_losses": adaptation_losses,
            "query_loss": query_loss.item(),
            "final_adaptation_loss": adaptation_losses[-1] if adaptation_losses else 0.0
        }
    
    def _compute_support_loss(self, support_data: List[Dict]) -> torch.Tensor:
        """Tính loss trên support set"""
        total_loss = 0.0
        num_samples = 0
        
        for sample in support_data:
            context = sample.get("context", "")
            response = sample.get("response", "")
            reward = sample.get("reward", 0.0)
            
            # Convert to tensors (simplified)
            context_tensor = torch.randn(1, 10, self.mann.input_size)  # Mock embedding
            target_tensor = torch.randn(1, 10, self.mann.output_size)  # Mock target
            
            # Forward pass
            output, retrieved_memories = self.mann(context_tensor)
            
            # Compute loss
            mse_loss = F.mse_loss(output, target_tensor)
            
            # Weight loss by reward
            weighted_loss = mse_loss * (1.0 + abs(reward))
            total_loss += weighted_loss
            num_samples += 1
        
        return total_loss / max(num_samples, 1)
    
    def _compute_query_loss(self, query_data: List[Dict]) -> torch.Tensor:
        """Tính loss trên query set"""
        return self._compute_support_loss(query_data)  # Simplified
    
    def store_episodic_experience(self, 
                                context: str,
                                response: str,
                                reward: float,
                                user_feedback: Optional[str] = None) -> None:
        """Store experience để sau này meta-learn"""
        # Convert to tensors (trong thực tế sẽ dùng proper embedding)
        context_tensor = torch.randn(10, self.mann.input_size)  # Mock
        response_tensor = torch.randn(10, self.mann.input_size)  # Mock
        
        # Store in MANN memory
        self.mann.store_experience(context_tensor, response_tensor, reward)
        
        # Store in experience buffer cho meta-learning
        experience = {
            "context": context,
            "response": response,
            "reward": reward,
            "user_feedback": user_feedback,
            "timestamp": torch.tensor(self.mann.timestep)
        }
        self.experience_buffer.append(experience)
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > 10000:
            self.experience_buffer = self.experience_buffer[-8000:]  # Keep recent 8000
    
    def generate_meta_learning_episodes(self, 
                                      num_episodes: int = 10,
                                      support_size: int = 5,
                                      query_size: int = 3) -> List[Tuple[List[Dict], List[Dict]]]:
        """Generate episodes cho meta-learning"""
        if len(self.experience_buffer) < support_size + query_size:
            return []
        
        episodes = []
        
        for _ in range(num_episodes):
            # Sample experiences
            sampled = np.random.choice(
                len(self.experience_buffer), 
                size=support_size + query_size, 
                replace=False
            )
            
            # Split into support và query
            support_indices = sampled[:support_size]
            query_indices = sampled[support_size:]
            
            support_set = [self.experience_buffer[i] for i in support_indices]
            query_set = [self.experience_buffer[i] for i in query_indices]
            
            episodes.append((support_set, query_set))
        
        return episodes
    
    def meta_learning_session(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Chạy meta-learning session"""
        if len(self.experience_buffer) < 20:
            return {"status": "insufficient_data", "buffer_size": len(self.experience_buffer)}
        
        # Generate episodes
        episodes = self.generate_meta_learning_episodes(num_episodes)
        
        session_results = {
            "episodes_results": [],
            "avg_query_loss": 0.0,
            "avg_adaptation_loss": 0.0,
            "memory_stats": self.mann.get_memory_statistics()
        }
        
        total_query_loss = 0.0
        total_adaptation_loss = 0.0
        
        for episode_idx, (support_set, query_set) in enumerate(episodes):
            episode_result = self.meta_train_episode(support_set, query_set)
            session_results["episodes_results"].append(episode_result)
            
            total_query_loss += episode_result["query_loss"]
            total_adaptation_loss += episode_result["final_adaptation_loss"]
            
            if episode_idx % 10 == 0:
                print(f"Episode {episode_idx}: Query Loss = {episode_result['query_loss']:.4f}")
        
        # Average results
        session_results["avg_query_loss"] = total_query_loss / num_episodes
        session_results["avg_adaptation_loss"] = total_adaptation_loss / num_episodes
        
        return session_results
    
    def select_relevant_memories(self, 
                                query: str, 
                                context: str = "",
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """Select relevant memories cho query"""
        # Mock query tensor
        query_tensor = torch.randn(1, self.mann.input_size)
        
        # Use MANN to retrieve memories
        with torch.no_grad():
            controller_output, _ = self.mann.controller(query_tensor.unsqueeze(1))
            memory_output, retrieved_info = self.mann.memory_interface.retrieve(
                controller_output.squeeze(1), self.mann.memory_bank, top_k=top_k
            )
        
        # Format results
        relevant_memories = []
        if retrieved_info:
            for info in retrieved_info[0]:  # First batch
                relevant_memories.append({
                    "memory_index": info["index"],
                    "similarity": info["similarity"],
                    "importance_weight": info["importance_weight"],
                    "usage_count": info["usage_count"]
                })
        
        return relevant_memories
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Thống kê tổng thể của system"""
        memory_stats = self.mann.get_memory_statistics()
        
        return {
            "meta_learning": self.meta_learning_stats,
            "memory_bank": memory_stats,
            "experience_buffer_size": len(self.experience_buffer),
            "total_experiences_stored": len(self.experience_buffer)
        }
    
    def save_system(self, filepath: str) -> None:
        """Lưu toàn bộ system"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            "mann_state_dict": self.mann.state_dict(),
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "meta_learning_stats": self.meta_learning_stats,
            "experience_buffer": self.experience_buffer[-1000:],  # Save recent 1000
            "memory_bank": [
                {
                    "key": entry.key.cpu(),
                    "value": entry.value.cpu(),
                    "usage_count": entry.usage_count,
                    "last_accessed": entry.last_accessed,
                    "importance_weight": entry.importance_weight
                } for entry in self.mann.memory_bank
            ]
        }, filepath)
    
    def load_system(self, filepath: str) -> bool:
        """Load system từ file"""
        if not os.path.exists(filepath):
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Load model state
            self.mann.load_state_dict(checkpoint["mann_state_dict"])
            self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
            
            # Load stats và data
            self.meta_learning_stats = checkpoint["meta_learning_stats"]
            self.experience_buffer = checkpoint["experience_buffer"]
            
            # Restore memory bank
            for i, entry_data in enumerate(checkpoint["memory_bank"]):
                if i < len(self.mann.memory_bank):
                    self.mann.memory_bank[i] = MemoryBankEntry(
                        key=entry_data["key"],
                        value=entry_data["value"],
                        usage_count=entry_data["usage_count"],
                        last_accessed=entry_data["last_accessed"],
                        importance_weight=entry_data["importance_weight"]
                    )
            
            return True
        except Exception as e:
            print(f"Lỗi khi load system: {e}")
            return False
