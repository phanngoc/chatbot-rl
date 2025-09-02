"""
Meta-learning với Episodic Memory
Học cách chọn lọc và kích hoạt trải nghiệm phù hợp
Implement Memory-Augmented Neural Network (MANN) patterns
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


@dataclass
class MemoryBankEntry:
    """Entry trong memory bank"""
    key: torch.Tensor  # Key vector để search
    value: torch.Tensor  # Value vector chứa information
    usage_count: int = 0
    last_accessed: int = 0  # Timestep cuối được access
    importance_weight: float = 1.0


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
        
        # Add device property
        self._device = None
    
    @property
    def device(self):
        """Get device của model"""
        if self._device is None:
            # Lấy device từ parameter đầu tiên
            for param in self.parameters():
                self._device = param.device
                break
            # Nếu không có parameters, default to CPU
            if self._device is None:
                self._device = torch.device('cpu')
        return self._device
    
    def to(self, device):
        """Move model đến device cụ thể"""
        super().to(device)
        self._device = device
        
        # Move memory bank tensors to device
        for entry in self.memory_bank:
            entry.key = entry.key.to(device)
            entry.value = entry.value.to(device)
        
        return self
    
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
        
        # Ensure tensors are on the same device as the model
        device = self.device
        key = key.detach().to(device)
        value = value.detach().to(device)
        
        self.memory_bank[lru_idx] = MemoryBankEntry(
            key=key,
            value=value,
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
        # Check if memory bank is empty
        if not memory_bank:
            batch_size = query_input.size(0)
            memory_dim = self.memory_dim if hasattr(self, 'memory_dim') else 128
            empty_output = torch.zeros(batch_size, memory_dim, device=query_input.device)
            return empty_output, []
        
        batch_size = query_input.size(0)
        
        # Generate query
        query = self.query_layer(query_input)  # (batch_size, memory_dim)
        
        # Calculate similarities với all memories
        similarities = []
        memory_values = []
        memory_info = []
        
        for i, entry in enumerate(memory_bank):
            # Ensure entry.key has correct shape
            if entry.key.dim() == 1:
                entry_key = entry.key.unsqueeze(0)  # (1, memory_dim)
            else:
                entry_key = entry.key
            
            # Ensure query has correct shape
            if query.dim() == 1:
                query_expanded = query.unsqueeze(0)  # (1, memory_dim)
            else:
                query_expanded = query
            
            # Cosine similarity
            key_norm = F.normalize(entry_key, p=2, dim=-1)
            query_norm = F.normalize(query_expanded, p=2, dim=-1)
            similarity = torch.matmul(query_norm, key_norm.t()).squeeze()  # (batch_size,)
            
            # Ensure similarity is 1D tensor
            if similarity.dim() == 0:
                similarity = similarity.unsqueeze(0)  # (1,)
            
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
        
        # Get top-k memories với proper error handling
        try:
            similarities_tensor = torch.stack(similarities, dim=-1)  # (batch_size, memory_size)
            
            # Ensure tensor has correct dimensions
            if similarities_tensor.dim() == 1:
                similarities_tensor = similarities_tensor.unsqueeze(0)  # (1, memory_size)
            
            actual_top_k = min(top_k, len(memory_bank))
            top_k_indices = torch.topk(similarities_tensor, actual_top_k, dim=-1).indices
            
            # Retrieve top-k memories
            retrieved_memories = []
            retrieved_info = []
            
            for batch_idx in range(batch_size):
                batch_memories = []
                batch_info = []
                
                for k_idx in range(actual_top_k):
                    # Ensure proper indexing
                    if top_k_indices.dim() == 1:
                        memory_idx = top_k_indices[k_idx].item()
                    else:
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
            
        except Exception as e:
            print(f"⚠️  Lỗi trong retrieve function: {e}")
            print(f"Debug info: batch_size={batch_size}, memory_bank_size={len(memory_bank)}")
            print(f"Similarities tensor shape: {[s.shape for s in similarities]}")
            
            # Fallback: return empty output
            memory_dim = self.memory_dim if hasattr(self, 'memory_dim') else 128
            fallback_output = torch.zeros(batch_size, memory_dim, device=query_input.device)
            return fallback_output, []


class MetaLearningEpisodicSystem:
    """
    Main system cho meta-learning với episodic memory
    Học cách select và activate relevant experiences
    Tích hợp với database persistence cho session-based memory
    """
    
    def __init__(self,
                 input_size: int = 768,
                 hidden_size: int = 256,
                 memory_size: int = 1000,
                 memory_dim: int = 128,
                 output_size: int = 768,
                 session_id: str = None):
        
        # Session management
        self.session_id = session_id
        self.logger = logging.getLogger("MetaLearningEpisodicSystem")
        
        # Initialize MANN
        self.mann = MemoryAugmentedNetwork(
            input_size, hidden_size, memory_size, memory_dim, output_size
        )
        
        # Ensure MANN has device property initialized
        if not hasattr(self.mann, 'device'):
            self.mann._device = torch.device('cpu')
        
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
        
        # Database integration (lazy import để tránh circular dependency)
        self._session_manager = None
        self._memory_loaded = False
        
        # Load memory bank từ database nếu có session_id
        if self.session_id:
            self._ensure_memory_loaded()
    
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
        """Select relevant memories cho query - Tối ưu hóa hiệu suất"""
        try:
            # Mock query tensor với pre-allocated memory
            device = self.mann.device
            query_tensor = torch.randn(1, self.mann.input_size, device=device)
            
            # Use MANN to retrieve memories với error handling
            with torch.no_grad():
                controller_output, _ = self.mann.controller(query_tensor.unsqueeze(1))
                
                # Ensure controller_output has correct shape
                if controller_output.dim() == 2:
                    controller_output = controller_output.unsqueeze(0)  # (1, seq_len, hidden_size)
                
                # Squeeze to get single timestep
                if controller_output.size(1) > 0:
                    timestep_output = controller_output[:, 0, :]  # (1, hidden_size)
                else:
                    # Fallback if no sequence
                    timestep_output = torch.zeros(1, controller_output.size(-1), device=device)
                
                memory_output, retrieved_info = self.mann.memory_interface.retrieve(
                    timestep_output, self.mann.memory_bank, top_k=top_k
                )
            
            # Optimized result formatting với list comprehension
            if retrieved_info and len(retrieved_info) > 0:
                batch_info = retrieved_info[0]  # First batch
                if batch_info:
                    return [
                        {
                            "memory_index": info.get("index", -1),
                            "similarity": info.get("similarity", 0.0),
                            "importance_weight": info.get("importance_weight", 1.0),
                            "usage_count": info.get("usage_count", 0)
                        }
                        for info in batch_info
                    ]
            
            return []
            
        except Exception as e:
            print(f"⚠️  Lỗi khi select relevant memories: {e}")
            print(f"Debug info: query='{query}', context='{context}', top_k={top_k}")
            print(f"MANN device: {getattr(self.mann, 'device', 'unknown')}")
            print(f"MANN input_size: {getattr(self.mann, 'input_size', 'unknown')}")
            print(f"Memory bank size: {len(getattr(self.mann, 'memory_bank', []))}")
            return []
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Thống kê tổng thể của system"""
        memory_stats = self.mann.get_memory_statistics()
        
        stats = {
            "meta_learning": self.meta_learning_stats,
            "memory_bank": memory_stats,
            "experience_buffer_size": len(self.experience_buffer),
            "total_experiences_stored": len(self.experience_buffer),
            "session_id": self.session_id,
            "memory_loaded_from_db": self._memory_loaded
        }
        
        # Add database stats nếu có session
        if self.session_id and self._get_session_manager():
            try:
                db_stats = self._get_session_manager().get_session_memory_stats(self.session_id)
                stats["database_memory_stats"] = db_stats
            except Exception as e:
                self.logger.warning(f"Failed to get database stats: {e}")
        
        return stats
    
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
    
    # === Database Integration Methods ===
    
    def _get_session_manager(self):
        """Lazy load session manager để tránh circular import"""
        if self._session_manager is None:
            try:
                from database.session_manager import get_session_manager
                self._session_manager = get_session_manager()
            except ImportError as e:
                self.logger.warning(f"Could not import session manager: {e}")
                return None
        return self._session_manager
    
    def set_session_id(self, session_id: str):
        """Set session ID và load memory từ database"""
        self.session_id = session_id
        self._memory_loaded = False
        if session_id:
            self._ensure_memory_loaded()
    
    def _ensure_memory_loaded(self):
        """Đảm bảo memory bank đã được load từ database"""
        if self._memory_loaded or not self.session_id:
            return
        
        session_manager = self._get_session_manager()
        if not session_manager:
            self.logger.warning("Session manager not available, using in-memory only")
            return
        
        try:
            # Load memory bank từ database
            memory_entries, timestep = session_manager.load_memory_bank_for_session(
                self.session_id, 
                device=self.mann.device
            )
            
            if memory_entries:
                # Update MANN memory bank
                self.mann.memory_bank = memory_entries
                self.mann.timestep = timestep
                self._memory_loaded = True
                
                self.logger.info(f"Loaded {len(memory_entries)} memory entries from database for session {self.session_id}")
            else:
                self.logger.info(f"No existing memory found for session {self.session_id}, starting fresh")
                self._memory_loaded = True
                
        except Exception as e:
            self.logger.error(f"Failed to load memory from database: {e}")
            # Continue với empty memory bank
            self._memory_loaded = True
    
    def save_memory_to_database(self, force_save: bool = False):
        """Lưu memory bank hiện tại vào database"""
        if not self.session_id:
            self.logger.warning("No session ID set, cannot save to database")
            return False
        
        session_manager = self._get_session_manager()
        if not session_manager:
            self.logger.warning("Session manager not available")
            return False
        
        try:
            # Ensure memory is loaded first
            self._ensure_memory_loaded()
            
            # Save current memory bank state
            session_manager.save_memory_bank_for_session(
                self.session_id,
                self.mann.memory_bank,
                self.mann.timestep
            )
            
            self.logger.info(f"Saved memory bank to database for session {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save memory to database: {e}")
            return False
    
    def auto_save_memory(self, save_interval: int = 10):
        """Auto-save memory bank theo interval (mỗi N operations)"""
        if hasattr(self, '_last_save_timestep'):
            steps_since_save = self.mann.timestep - self._last_save_timestep
            if steps_since_save >= save_interval:
                if self.save_memory_to_database():
                    self._last_save_timestep = self.mann.timestep
        else:
            self._last_save_timestep = 0
            self.save_memory_to_database()
    
    def store_episodic_experience_with_autosave(self, 
                                               context: str,
                                               response: str,
                                               reward: float,
                                               user_feedback: Optional[str] = None) -> None:
        """Store experience và auto-save khi cần"""
        # Call original method
        self.store_episodic_experience(context, response, reward, user_feedback)
        
        # Auto-save theo interval
        self.auto_save_memory(save_interval=5)  # Save mỗi 5 experiences
    
    def get_memory_for_session(self, session_id: str = None) -> List[MemoryBankEntry]:
        """Lấy memory bank cho session cụ thể"""
        if session_id is None:
            session_id = self.session_id
        
        if not session_id:
            return self.mann.memory_bank
        
        session_manager = self._get_session_manager()
        if not session_manager:
            return self.mann.memory_bank
        
        try:
            memory_entries, _ = session_manager.load_memory_bank_for_session(
                session_id, 
                device=self.mann.device
            )
            return memory_entries
        except Exception as e:
            self.logger.error(f"Failed to load memory for session {session_id}: {e}")
            return []
    
    def clear_session_memory(self, session_id: str = None):
        """Clear memory cho session (for testing/reset)"""
        if session_id is None:
            session_id = self.session_id
        
        if session_id:
            # Clear trong database
            session_manager = self._get_session_manager()
            if session_manager:
                try:
                    # Save empty memory bank
                    session_manager.save_memory_bank_for_session(session_id, [], 0)
                    self.logger.info(f"Cleared memory for session {session_id}")
                except Exception as e:
                    self.logger.error(f"Failed to clear session memory: {e}")
        
        # Clear current memory nếu đây là current session
        if session_id == self.session_id or session_id is None:
            # Reset memory bank
            memory_size = len(self.mann.memory_bank)
            memory_dim = self.mann.memory_dim
            device = self.mann.device
            
            self.mann.memory_bank = [
                MemoryBankEntry(
                    key=torch.randn(memory_dim, device=device),
                    value=torch.randn(memory_dim, device=device)
                ) for _ in range(memory_size)
            ]
            self.mann.timestep = 0
            self._memory_loaded = True
