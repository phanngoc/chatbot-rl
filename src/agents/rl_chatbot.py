"""
RL Chatbot Agent tích hợp tất cả các thành phần
"""

import torch
import torch.nn as nn
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import logging

# Import các components đã implement
from ..core.experience_replay import ExperienceReplayBuffer, Experience, ExperienceReplayTrainer
from ..memory.retrieval_memory import RetrievalAugmentedMemory
from ..memory.consolidation import MemoryConsolidationSystem
from ..core.ewc import MultiTaskEWC
from ..core.meta_learning import MetaLearningEpisodicSystem
from ..core.temporal_weighting import TemporalWeightingSystem


class RLChatbotModel(nn.Module):
    """Neural network model cho RL Chatbot"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 hidden_size: int = 768,
                 memory_dim: int = 256):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        
        # Memory integration layers
        self.memory_projection = nn.Linear(memory_dim, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Response generation
        self.response_head = nn.Linear(hidden_size, self.tokenizer.vocab_size)
        
        # Value head cho RL
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                memory_context: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass với memory integration"""
        
        # Base model forward
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Memory integration
        if memory_context is not None:
            # Project memory to hidden size
            memory_projected = self.memory_projection(memory_context)  # (batch_size, memory_len, hidden_size)
            
            # Attention between hidden states và memory
            attended_output, attention_weights = self.attention(
                hidden_states, memory_projected, memory_projected
            )
            
            # Combine với residual connection
            hidden_states = hidden_states + self.dropout(attended_output)
        
        # Generate response logits
        response_logits = self.response_head(hidden_states)
        
        # Generate value estimate
        value_estimate = self.value_head(hidden_states.mean(dim=1))  # Pool over sequence
        
        return {
            "response_logits": response_logits,
            "value_estimate": value_estimate,
            "hidden_states": hidden_states,
            "attention_weights": attention_weights if memory_context is not None else None
        }
    
    def generate_response(self, 
                         input_text: str,
                         memory_context: torch.Tensor = None,
                         max_length: int = 100,
                         temperature: float = 0.8) -> str:
        """Generate response text"""
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                memory_context=memory_context
            )
            
            # Generate response using sampling
            response_logits = outputs["response_logits"][:, -1, :]  # Last token logits
            
            # Apply temperature
            response_logits = response_logits / temperature
            
            # Sample next token
            probs = torch.softmax(response_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # For simplicity, just return a mock response
            # Trong thực tế sẽ implement full autoregressive generation
            response = "Đây là response được generate từ RL Chatbot."
        
        return response


class RLChatbotAgent:
    """Main RL Chatbot Agent tích hợp tất cả components"""
    
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "cpu",
                 config: Dict[str, Any] = None):
        
        self.device = device
        self.config = config or {}
        
        # Initialize model
        self.model = RLChatbotModel(model_name).to(device)
        
        # Initialize all RL components
        self._initialize_components()
        
        # Conversation state
        self.current_conversation_id = None
        self.conversation_history = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_interactions": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "avg_response_time": 0.0,
            "memory_retrievals": 0,
            "consolidation_runs": 0
        }
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _initialize_components(self):
        """Initialize tất cả RL components"""
        
        # Experience Replay
        self.experience_buffer = ExperienceReplayBuffer(
            max_size=self.config.get("experience_buffer_size", 10000),
            save_path=self.config.get("experience_save_path", "data/experience_buffer.pkl")
        )
        
        # Retrieval-Augmented Memory
        self.retrieval_memory = RetrievalAugmentedMemory(
            store_type=self.config.get("memory_store_type", "chroma"),
            max_memories=self.config.get("max_memories", 5000)
        )
        
        # Memory Consolidation
        self.consolidation_system = MemoryConsolidationSystem(
            consolidation_threshold=self.config.get("consolidation_threshold", 100),
            consolidation_interval_hours=self.config.get("consolidation_interval", 24)
        )
        
        # EWC System
        self.ewc_system = MultiTaskEWC(
            model=self.model,
            ewc_lambda=self.config.get("ewc_lambda", 1000.0)
        )
        
        # Meta-learning System
        self.meta_learning_system = MetaLearningEpisodicSystem(
            input_size=768,
            hidden_size=256,
            memory_size=self.config.get("meta_memory_size", 1000),
            memory_dim=128,
            output_size=768
        )
        
        # Temporal Weighting System
        self.temporal_weighting = TemporalWeightingSystem(
            decay_function=self.config.get("decay_function", "exponential"),
            decay_params=self.config.get("decay_params", {"decay_rate": 0.05}),
            update_interval_hours=self.config.get("weight_update_interval", 6)
        )
        
        # Experience Replay Trainer
        self.replay_trainer = ExperienceReplayTrainer(
            model=self.model,
            buffer=self.experience_buffer,
            learning_rate=self.config.get("learning_rate", 1e-4)
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("RLChatbot")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def start_conversation(self, user_id: str = None) -> str:
        """Bắt đầu conversation mới"""
        self.current_conversation_id = str(uuid.uuid4())
        self.conversation_history = []
        
        self.logger.info(f"Bắt đầu conversation mới: {self.current_conversation_id}")
        
        return self.current_conversation_id
    
    def process_message(self, 
                       user_message: str,
                       context: str = "",
                       user_feedback: Optional[float] = None) -> Dict[str, Any]:
        """Process user message và generate response"""
        
        start_time = datetime.now()
        
        if not self.current_conversation_id:
            self.start_conversation()
        
        # 1. Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(user_message, context)
        
        # 2. Generate response với memory context
        response = self._generate_response_with_memory(user_message, relevant_memories)
        
        # 3. Store experience
        experience_id = self._store_experience(user_message, response, context, user_feedback)
        
        # 4. Update conversation history
        self.conversation_history.append({
            "user_message": user_message,
            "bot_response": response,
            "timestamp": start_time,
            "experience_id": experience_id,
            "relevant_memories": len(relevant_memories)
        })
        
        # 5. Update performance metrics
        self._update_performance_metrics(start_time, user_feedback)
        
        # 6. Check if consolidation needed
        self._check_and_run_consolidation()
        
        # 7. Periodic weight updates
        self.temporal_weighting.batch_update_weights()
        
        return {
            "response": response,
            "conversation_id": self.current_conversation_id,
            "experience_id": experience_id,
            "relevant_memories_count": len(relevant_memories),
            "response_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "memory_stats": self._get_memory_stats()
        }
    
    def _retrieve_relevant_memories(self, 
                                   user_message: str, 
                                   context: str = "") -> List[Dict[str, Any]]:
        """Retrieve relevant memories cho response generation"""
        
        # Combine user message và context
        query = f"{context} {user_message}".strip()
        
        # Retrieve từ different memory systems
        memories = []
        
        # 1. Retrieval-Augmented Memory
        ram_memories = self.retrieval_memory.retrieve_relevant_memories(
            query, top_k=3
        )
        memories.extend(ram_memories)
        
        # 2. Meta-learning system
        meta_memories = self.meta_learning_system.select_relevant_memories(
            query, context, top_k=2
        )
        memories.extend(meta_memories)
        
        # 3. Temporal weighted experiences
        weighted_experiences = self.temporal_weighting.get_weighted_experiences(
            top_k=5, min_weight=0.3
        )
        
        for exp in weighted_experiences[:2]:  # Take top 2
            memories.append({
                "content": exp.content,
                "context": exp.context,
                "weight": exp.get_combined_weight(),
                "source": "temporal_weighting"
            })
        
        self.performance_metrics["memory_retrievals"] += 1
        
        return memories
    
    def _generate_response_with_memory(self, 
                                     user_message: str,
                                     memories: List[Dict[str, Any]]) -> str:
        """Generate response sử dụng retrieved memories"""
        
        # Prepare memory context tensor
        memory_context = None
        if memories:
            # Trong thực tế sẽ convert memories thành proper embeddings
            # Ở đây chỉ mock một tensor
            memory_dim = 256
            memory_context = torch.randn(1, len(memories), memory_dim)
        
        # Generate response
        response = self.model.generate_response(
            user_message, 
            memory_context=memory_context,
            temperature=self.config.get("temperature", 0.8)
        )
        
        # Enhance response với memory information (simplified)
        if memories:
            memory_info = f" [Sử dụng {len(memories)} memories liên quan]"
            response += memory_info
        
        return response
    
    def _store_experience(self, 
                         user_message: str,
                         bot_response: str,
                         context: str = "",
                         user_feedback: Optional[float] = None) -> str:
        """Store experience trong tất cả memory systems"""
        
        experience_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Calculate reward từ feedback (nếu có)
        if user_feedback is not None:
            reward = user_feedback
        else:
            # Default neutral reward
            reward = 0.0
        
        # 1. Store trong Experience Replay Buffer
        experience = Experience(
            state=f"{context} {user_message}".strip(),
            action=bot_response,
            reward=reward,
            next_state="",  # Will be filled by next interaction
            timestamp=timestamp,
            conversation_id=self.current_conversation_id,
            user_feedback=str(user_feedback) if user_feedback is not None else None
        )
        self.experience_buffer.add_experience(experience)
        
        # 2. Store trong Retrieval Memory
        memory_id = self.retrieval_memory.add_memory(
            content=bot_response,
            context=f"{context} {user_message}".strip(),
            tags=["conversation", "response"],
            importance_score=1.0 + abs(reward),
            metadata={
                "conversation_id": self.current_conversation_id,
                "user_feedback": user_feedback,
                "experience_id": experience_id
            }
        )
        
        # 3. Store trong Meta-learning System
        self.meta_learning_system.store_episodic_experience(
            context=f"{context} {user_message}".strip(),
            response=bot_response,
            reward=reward,
            user_feedback=str(user_feedback) if user_feedback is not None else None
        )
        
        # 4. Store trong Temporal Weighting System
        self.temporal_weighting.add_experience(
            experience_id=experience_id,
            content=bot_response,
            context=f"{context} {user_message}".strip(),
            reward=reward,
            tags=["conversation", "response"],
            source="user_interaction"
        )
        
        return experience_id
    
    def _update_performance_metrics(self, 
                                  start_time: datetime,
                                  user_feedback: Optional[float] = None):
        """Update performance metrics"""
        
        # Response time
        response_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics["avg_response_time"] = (
            (self.performance_metrics["avg_response_time"] * self.performance_metrics["total_interactions"] + 
             response_time) / (self.performance_metrics["total_interactions"] + 1)
        )
        
        # Interaction count
        self.performance_metrics["total_interactions"] += 1
        
        # Feedback tracking
        if user_feedback is not None:
            if user_feedback > 0.5:
                self.performance_metrics["positive_feedback"] += 1
            elif user_feedback < -0.5:
                self.performance_metrics["negative_feedback"] += 1
    
    def _check_and_run_consolidation(self):
        """Kiểm tra và chạy memory consolidation nếu cần"""
        
        num_new_memories = len(self.experience_buffer.buffer)
        
        if self.consolidation_system.should_consolidate(num_new_memories):
            self.logger.info("Chạy memory consolidation...")
            
            # Prepare memories cho consolidation
            experiences_for_consolidation = []
            for exp in list(self.experience_buffer.buffer)[-200:]:  # Recent 200
                experiences_for_consolidation.append({
                    "id": str(uuid.uuid4()),
                    "content": exp.action,
                    "context": exp.state,
                    "reward": exp.reward,
                    "timestamp": exp.timestamp
                })
            
            # Run consolidation
            consolidation_results = self.consolidation_system.consolidate_memories(
                experiences_for_consolidation,
                method="all"
            )
            
            self.performance_metrics["consolidation_runs"] += 1
            self.logger.info(f"Consolidation completed: {consolidation_results}")
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Lấy statistics từ tất cả memory systems"""
        
        return {
            "experience_buffer": self.experience_buffer.get_statistics(),
            "retrieval_memory": self.retrieval_memory.get_memory_statistics(),
            "temporal_weighting": self.temporal_weighting.get_weight_statistics(),
            "meta_learning": self.meta_learning_system.get_system_statistics()
        }
    
    def provide_feedback(self, 
                        experience_id: str,
                        feedback_score: float,
                        feedback_text: str = "") -> bool:
        """Provide feedback cho một experience cụ thể"""
        
        success = True
        
        # Update trong Experience Buffer
        for conv_id, experiences in self.experience_buffer.conversation_history.items():
            for i, exp in enumerate(experiences):
                if hasattr(exp, 'id') and exp.id == experience_id:
                    success &= self.experience_buffer.update_experience_reward(
                        conv_id, i, feedback_score, feedback_text
                    )
        
        # Update trong Temporal Weighting System
        success &= self.temporal_weighting.update_experience_feedback(
            experience_id, feedback_score
        )
        
        # Trigger learning từ feedback
        if abs(feedback_score) > 0.5:  # Strong feedback
            self._trigger_learning_from_feedback(experience_id, feedback_score)
        
        return success
    
    def _trigger_learning_from_feedback(self, 
                                      experience_id: str,
                                      feedback_score: float):
        """Trigger learning khi có strong feedback"""
        
        # Experience replay training
        if len(self.experience_buffer.buffer) >= 32:
            training_results = self.replay_trainer.replay_training_step(
                batch_size=16,
                num_epochs=1
            )
            self.logger.info(f"Feedback-triggered training: {training_results}")
        
        # Meta-learning session (nếu có đủ data)
        if len(self.meta_learning_system.experience_buffer) >= 20:
            meta_results = self.meta_learning_system.meta_learning_session(num_episodes=5)
            self.logger.info(f"Meta-learning session: {meta_results.get('avg_query_loss', 'N/A')}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Lấy summary của conversation hiện tại"""
        
        if not self.current_conversation_id:
            return {"error": "Không có conversation nào đang active"}
        
        return {
            "conversation_id": self.current_conversation_id,
            "total_exchanges": len(self.conversation_history),
            "start_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None,
            "total_memories_used": sum(h["relevant_memories"] for h in self.conversation_history),
            "conversation_history": self.conversation_history[-10:]  # Recent 10 exchanges
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Lấy status tổng thể của system"""
        
        return {
            "model_info": {
                "model_name": "RLChatbot",
                "device": self.device,
                "total_parameters": sum(p.numel() for p in self.model.parameters())
            },
            "performance_metrics": self.performance_metrics,
            "memory_systems": self._get_memory_stats(),
            "current_conversation": self.current_conversation_id,
            "system_health": {
                "experience_buffer_utilization": len(self.experience_buffer.buffer) / self.experience_buffer.max_size * 100,
                "memory_consolidation_status": "active" if self.consolidation_system else "inactive",
                "meta_learning_episodes": self.meta_learning_system.meta_learning_stats.get("episodes_trained", 0)
            }
        }
    
    def save_agent_state(self, filepath: str) -> bool:
        """Lưu state của agent"""
        
        try:
            # Save model
            model_path = filepath.replace('.json', '_model.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, model_path)
            
            # Save all systems
            self.experience_buffer.save_buffer()
            self.retrieval_memory.save_memories(filepath.replace('.json', '_retrieval_memories.json'))
            self.consolidation_system.save_consolidated_knowledge(filepath.replace('.json', '_consolidated_knowledge.json'))
            self.ewc_system.save_ewc_data(filepath.replace('.json', '_ewc_data.pkl'))
            self.meta_learning_system.save_system(filepath.replace('.json', '_meta_learning.pt'))
            self.temporal_weighting.save_system(filepath.replace('.json', '_temporal_weights.json'))
            
            # Save agent state
            agent_state = {
                "current_conversation_id": self.current_conversation_id,
                "conversation_history": [
                    {**h, "timestamp": h["timestamp"].isoformat()} 
                    for h in self.conversation_history
                ],
                "performance_metrics": self.performance_metrics,
                "config": self.config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(agent_state, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Agent state saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi save agent state: {e}")
            return False
    
    def load_agent_state(self, filepath: str) -> bool:
        """Load state của agent"""
        
        try:
            # Load model
            model_path = filepath.replace('.json', '_model.pt')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.config.update(checkpoint.get('config', {}))
            
            # Load all systems
            self.experience_buffer.load_buffer()
            self.retrieval_memory.load_memories(filepath.replace('.json', '_retrieval_memories.json'))
            self.ewc_system.load_ewc_data(filepath.replace('.json', '_ewc_data.pkl'))
            self.meta_learning_system.load_system(filepath.replace('.json', '_meta_learning.pt'))
            self.temporal_weighting.load_system(filepath.replace('.json', '_temporal_weights.json'))
            
            # Load agent state
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    agent_state = json.load(f)
                
                self.current_conversation_id = agent_state.get("current_conversation_id")
                self.performance_metrics = agent_state.get("performance_metrics", {})
                
                # Restore conversation history
                self.conversation_history = []
                for h in agent_state.get("conversation_history", []):
                    h_copy = h.copy()
                    h_copy["timestamp"] = datetime.fromisoformat(h["timestamp"])
                    self.conversation_history.append(h_copy)
            
            self.logger.info(f"Agent state loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi load agent state: {e}")
            return False
