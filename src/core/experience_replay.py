"""
Experience Replay System cho Chatbot RL
Lưu trữ và replay các trải nghiệm để tránh catastrophic forgetting
"""

import random
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import numpy as np
import torch
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Experience:
    """Đại diện cho một trải nghiệm trong conversation"""
    state: str  # Context/input từ user
    action: str  # Response từ chatbot
    reward: float  # Feedback score (1.0 = positive, -1.0 = negative, 0.0 = neutral)
    next_state: str  # Context sau response
    timestamp: datetime
    conversation_id: str
    user_feedback: Optional[str] = None
    importance_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience thành dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Tạo Experience từ dictionary"""
        return cls(**data)


class ExperienceReplayBuffer:
    """Buffer lưu trữ và quản lý các trải nghiệm"""
    
    def __init__(self, 
                 max_size: int = 10000,
                 min_size_for_replay: int = 100,
                 save_path: str = "data/experience_buffer.pkl"):
        self.max_size = max_size
        self.min_size_for_replay = min_size_for_replay
        self.save_path = save_path
        self.buffer: deque = deque(maxlen=max_size)
        self.conversation_history: Dict[str, List[Experience]] = {}
        
        # Load existing buffer if exists
        self.load_buffer()
    
    def add_experience(self, experience: Experience) -> None:
        """Thêm trải nghiệm mới vào buffer"""
        self.buffer.append(experience)
        
        # Cập nhật conversation history
        conv_id = experience.conversation_id
        if conv_id not in self.conversation_history:
            self.conversation_history[conv_id] = []
        self.conversation_history[conv_id].append(experience)
        
        # Auto-save periodically
        if len(self.buffer) % 100 == 0:
            self.save_buffer()
    
    def sample_batch(self, batch_size: int = 32, 
                    prioritize_recent: bool = True,
                    prioritize_important: bool = True) -> List[Experience]:
        """Sample một batch experiences để training"""
        if len(self.buffer) < self.min_size_for_replay:
            return []
        
        available_size = min(batch_size, len(self.buffer))
        
        if prioritize_recent or prioritize_important:
            # Weighted sampling
            weights = self._calculate_sampling_weights(prioritize_recent, prioritize_important)
            indices = np.random.choice(
                len(self.buffer), 
                size=available_size, 
                replace=False,
                p=weights
            )
            return [self.buffer[i] for i in indices]
        else:
            # Random sampling
            return random.sample(list(self.buffer), available_size)
    
    def _calculate_sampling_weights(self, 
                                  prioritize_recent: bool = True,
                                  prioritize_important: bool = True) -> np.ndarray:
        """Tính trọng số cho việc sampling"""
        weights = np.ones(len(self.buffer))
        
        if prioritize_recent:
            # Trọng số tăng theo thời gian (experience gần đây có trọng số cao hơn)
            for i, exp in enumerate(self.buffer):
                days_old = (datetime.now() - exp.timestamp).days
                weights[i] *= np.exp(-days_old * 0.1)  # Decay factor
        
        if prioritize_important:
            # Trọng số dựa trên importance weight và reward
            for i, exp in enumerate(self.buffer):
                weights[i] *= exp.importance_weight * (1 + abs(exp.reward))
        
        # Normalize weights
        weights = weights / weights.sum()
        return weights
    
    def get_conversation_experiences(self, conversation_id: str) -> List[Experience]:
        """Lấy tất cả experiences của một conversation"""
        return self.conversation_history.get(conversation_id, [])
    
    def update_experience_reward(self, 
                                conversation_id: str, 
                                experience_index: int, 
                                new_reward: float,
                                user_feedback: str = None) -> bool:
        """Cập nhật reward cho một experience cụ thể"""
        if conversation_id not in self.conversation_history:
            return False
        
        conv_experiences = self.conversation_history[conversation_id]
        if experience_index >= len(conv_experiences):
            return False
        
        # Update experience
        experience = conv_experiences[experience_index]
        experience.reward = new_reward
        if user_feedback:
            experience.user_feedback = user_feedback
        
        # Update importance weight based on feedback
        if abs(new_reward) > 0.5:  # Strong feedback
            experience.importance_weight = min(experience.importance_weight * 1.5, 3.0)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Thống kê về buffer"""
        if not self.buffer:
            return {"total_experiences": 0}
        
        rewards = [exp.reward for exp in self.buffer]
        return {
            "total_experiences": len(self.buffer),
            "total_conversations": len(self.conversation_history),
            "avg_reward": np.mean(rewards),
            "positive_experiences": sum(1 for r in rewards if r > 0),
            "negative_experiences": sum(1 for r in rewards if r < 0),
            "neutral_experiences": sum(1 for r in rewards if r == 0),
            "buffer_utilization": len(self.buffer) / self.max_size * 100
        }
    
    def save_buffer(self) -> None:
        """Lưu buffer xuống file"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'conversation_history': self.conversation_history
            }, f)
    
    def load_buffer(self) -> None:
        """Load buffer từ file"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                    self.buffer = deque(data['buffer'], maxlen=self.max_size)
                    self.conversation_history = data['conversation_history']
            except Exception as e:
                print(f"Không thể load buffer: {e}")
    
    def clear_old_experiences(self, days_threshold: int = 30) -> int:
        """Xóa các experiences cũ hơn threshold"""
        current_time = datetime.now()
        old_count = 0
        
        # Filter buffer
        new_buffer = deque(maxlen=self.max_size)
        for exp in self.buffer:
            days_old = (current_time - exp.timestamp).days
            if days_old <= days_threshold:
                new_buffer.append(exp)
            else:
                old_count += 1
        
        self.buffer = new_buffer
        
        # Clean conversation history
        for conv_id in list(self.conversation_history.keys()):
            self.conversation_history[conv_id] = [
                exp for exp in self.conversation_history[conv_id]
                if (current_time - exp.timestamp).days <= days_threshold
            ]
            if not self.conversation_history[conv_id]:
                del self.conversation_history[conv_id]
        
        return old_count


class ExperienceReplayTrainer:
    """Trainer sử dụng Experience Replay để training model"""
    
    def __init__(self, 
                 model,
                 buffer: ExperienceReplayBuffer,
                 learning_rate: float = 1e-4):
        self.model = model
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
    
    def replay_training_step(self, 
                           batch_size: int = 32,
                           num_epochs: int = 1) -> Dict[str, float]:
        """Thực hiện một bước training với experience replay"""
        batch = self.buffer.sample_batch(batch_size)
        if not batch:
            return {"loss": 0.0, "batch_size": 0}
        
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Convert experiences to training data
            states, actions, rewards = self._prepare_training_data(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(states)
            
            # Calculate loss (simplified - trong thực tế sẽ phức tạp hơn)
            target_values = torch.tensor(rewards, dtype=torch.float32)
            loss = self.criterion(predictions.squeeze(), target_values)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_epochs
        return {
            "loss": avg_loss,
            "batch_size": len(batch),
            "epochs": num_epochs
        }
    
    def _prepare_training_data(self, batch: List[Experience]) -> Tuple[List[str], List[str], List[float]]:
        """Chuẩn bị data từ batch experiences"""
        states = [exp.state for exp in batch]
        actions = [exp.action for exp in batch]
        rewards = [exp.reward for exp in batch]
        
        return states, actions, rewards
    
    def evaluate_on_recent_experiences(self, num_recent: int = 100) -> Dict[str, float]:
        """Đánh giá model trên các experiences gần đây"""
        if len(self.buffer.buffer) < num_recent:
            recent_experiences = list(self.buffer.buffer)
        else:
            recent_experiences = list(self.buffer.buffer)[-num_recent:]
        
        if not recent_experiences:
            return {"accuracy": 0.0, "avg_reward": 0.0}
        
        # Simplified evaluation
        total_reward = sum(exp.reward for exp in recent_experiences)
        avg_reward = total_reward / len(recent_experiences)
        
        # Calculate "accuracy" based on positive rewards
        positive_count = sum(1 for exp in recent_experiences if exp.reward > 0)
        accuracy = positive_count / len(recent_experiences)
        
        return {
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "total_experiences": len(recent_experiences)
        }
