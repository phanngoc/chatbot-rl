"""
Episodic Memory Consolidation System - OpenAI Edition
Chuyển đổi từ episodic memory sang semantic memory sử dụng OpenAI API
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import pickle
import os
from openai import OpenAI
import tiktoken
import time
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator để profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        profiler.disable()
        
        # Log performance metrics
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

@dataclass
class ConsolidatedKnowledge:
    """Đại diện cho knowledge đã được consolidate"""
    id: str
    summary: str
    source_memories: List[str]  # IDs của episodic memories gốc
    confidence_score: float
    consolidation_method: str  # "summarization", "graph", "distillation"
    created_at: datetime
    access_count: int = 0
    semantic_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "summary": self.summary,
            "source_memories": self.source_memories,
            "confidence_score": self.confidence_score,
            "consolidation_method": self.consolidation_method,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "semantic_embedding": self.semantic_embedding
        }
        return data


class LLMSummarizer:
    """Sử dụng OpenAI để tóm tắt nhiều episodic memories thành knowledge chunks"""
    
    def __init__(self, openai_model: str = "gpt-3.5-turbo", api_key: str = None):        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = openai_model
    
    def summarize_memories(self, 
                          memories: List[Dict[str, Any]], 
                          max_tokens: int = 200) -> str:
        """Tóm tắt một nhóm memories thành knowledge summary"""
        # Prepare input text
        memory_texts = []
        for memory in memories:
            text = f"Context: {memory.get('context', '')}\nContent: {memory.get('content', '')}"
            memory_texts.append(text)
        
        combined_text = "\n---\n".join(memory_texts)
        
        # Create summarization prompt
        prompt = f"""Tóm tắt các trải nghiệm hội thoại sau thành kiến thức hữu ích:

{combined_text}

Tóm tắt:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là một AI chuyên tóm tắt và phân tích dữ liệu hội thoại."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Lỗi khi tóm tắt với OpenAI: {e}")
            return "Không thể tóm tắt memories."
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Trích xuất các concept chính từ text sử dụng OpenAI"""
        prompt = f"""Trích xuất các khái niệm và chủ đề chính từ văn bản sau. Trả về danh sách tối đa 10 khái niệm, mỗi khái niệm trên một dòng:

{text}

Các khái niệm chính:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là một AI chuyên trích xuất khái niệm từ văn bản."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            concepts_text = response.choices[0].message.content.strip()
            concepts = [concept.strip() for concept in concepts_text.split('\n') if concept.strip()]
            return concepts[:10]
            
        except Exception as e:
            print(f"Lỗi khi trích xuất concepts: {e}")
            # Fallback to simple extraction
            words = text.lower().split()
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'là', 'của', 'và', 'có', 'được', 'này', 'đó'}
            concepts = [word for word in words if len(word) > 3 and word not in stopwords]
            return list(set(concepts))[:10]


class KnowledgeGraph:
    """Knowledge Graph để tích hợp consolidated knowledge - Tối ưu hóa hiệu suất"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings: Dict[str, List[float]] = {}
        self.concept_memories: Dict[str, List[str]] = defaultdict(list)
        
        # Performance optimization: caching và batch operations
        self._concept_cache = {}  # Cache concept lookups
        self._batch_size = 100    # Batch size cho operations
        self._pending_operations = []  # Queue cho batch operations
    
    def add_knowledge(self, 
                     consolidated_knowledge: ConsolidatedKnowledge,
                     concepts: List[str]) -> None:
        """Thêm consolidated knowledge vào graph - Tối ưu hóa hiệu suất"""
        if not concepts:
            return
            
        knowledge_id = consolidated_knowledge.id
        
        # Batch add knowledge node
        self.graph.add_node(
            knowledge_id,
            type="knowledge",
            summary=consolidated_knowledge.summary,
            confidence=consolidated_knowledge.confidence_score,
            method=consolidated_knowledge.consolidation_method
        )
        
        # Pre-compute concept IDs và batch operations
        concept_nodes = []
        edges_to_add = []
        
        for concept in concepts:
            concept_id = f"concept_{concept.replace(' ', '_')}"
            concept_nodes.append((concept_id, concept))
            edges_to_add.append((knowledge_id, concept_id, {"relation": "contains"}))
        
        # Batch add concept nodes
        for concept_id, concept_name in concept_nodes:
            if concept_id not in self.graph:
                self.graph.add_node(concept_id, type="concept", name=concept_name)
        
        # Batch add edges
        self.graph.add_edges_from(edges_to_add)
        
        # Batch update concept memories
        source_memories = consolidated_knowledge.source_memories
        for concept in concepts:
            self.concept_memories[concept].extend(source_memories)
    
    def find_related_knowledge(self, 
                             query_concepts: List[str], 
                             max_depth: int = 2) -> List[Dict[str, Any]]:
        """Tìm knowledge liên quan đến query concepts - Tối ưu hóa hiệu suất"""
        if not query_concepts:
            return []
        
        # Pre-compute concept IDs
        concept_ids = [f"concept_{concept.replace(' ', '_')}" for concept in query_concepts]
        
        # Batch collect knowledge nodes
        knowledge_nodes = set()
        concept_to_knowledge = {}
        
        for concept_id, concept_name in zip(concept_ids, query_concepts):
            if concept_id in self.graph:
                predecessors = list(self.graph.predecessors(concept_id))
                knowledge_nodes.update(predecessors)
                concept_to_knowledge[concept_id] = concept_name
        
        # Batch extract node data
        related_knowledge = []
        for knowledge_id in knowledge_nodes:
            if self.graph.nodes[knowledge_id]['type'] == 'knowledge':
                # Find matching concept
                matching_concept = None
                for concept_id in concept_ids:
                    if concept_id in concept_to_knowledge and knowledge_id in self.graph.predecessors(concept_id):
                        matching_concept = concept_to_knowledge[concept_id]
                        break
                
                if matching_concept:
                    knowledge_data = {
                        'id': knowledge_id,
                        'summary': self.graph.nodes[knowledge_id]['summary'],
                        'confidence': self.graph.nodes[knowledge_id]['confidence'],
                        'method': self.graph.nodes[knowledge_id]['method'],
                        'matching_concept': matching_concept
                    }
                    related_knowledge.append(knowledge_data)
        
        # Optimized sorting với key function caching
        if related_knowledge:
            return sorted(related_knowledge, key=lambda x: x['confidence'], reverse=True)
        
        return []
    
    def get_concept_relationships(self, concept: str) -> Dict[str, List[str]]:
        """Lấy relationships của một concept"""
        concept_id = f"concept_{concept.replace(' ', '_')}"
        relationships = {
            'related_concepts': [],
            'knowledge_items': [],
            'co_occurring_concepts': []
        }
        
        if concept_id not in self.graph:
            return relationships
        
        # Get directly connected knowledge
        for knowledge_id in self.graph.predecessors(concept_id):
            if self.graph.nodes[knowledge_id]['type'] == 'knowledge':
                relationships['knowledge_items'].append(knowledge_id)
                
                # Find co-occurring concepts
                for related_concept_id in self.graph.successors(knowledge_id):
                    if (related_concept_id != concept_id and 
                        self.graph.nodes[related_concept_id]['type'] == 'concept'):
                        concept_name = self.graph.nodes[related_concept_id]['name']
                        relationships['co_occurring_concepts'].append(concept_name)
        
        return relationships
    
    def save_graph(self, filepath: str) -> None:
        """Lưu knowledge graph"""
        # Tạo thư mục nếu cần thiết
        directory = os.path.dirname(filepath)
        if directory:  # Chỉ tạo thư mục nếu có đường dẫn
            os.makedirs(directory, exist_ok=True)
            
        data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True)),
            'concept_embeddings': self.concept_embeddings,
            'concept_memories': dict(self.concept_memories)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_graph(self, filepath: str) -> bool:
        """Load knowledge graph"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(data['nodes'].items())
            self.graph.add_edges_from(data['edges'])
            self.concept_embeddings = data['concept_embeddings']
            self.concept_memories = defaultdict(list, data['concept_memories'])
            
            return True
        except Exception as e:
            print(f"Lỗi khi load graph: {e}")
            return False


class ModelDistillation:
    """
    Knowledge Distillation cho Episodic Memory System
    Thực hiện đúng chuẩn Teacher-Student architecture với soft targets và temperature scaling
    """
    
    def __init__(self, 
                 learning_rate: float = 1e-4,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 openai_model: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002",
                 api_key: str = None):
        
        # Knowledge Distillation parameters
        self.temperature = temperature  # Temperature scaling cho soft targets
        self.alpha = alpha  # Weight cho distillation loss  
        self.beta = beta    # Weight cho student loss (alpha + beta = 1.0)

        # OpenAI setup
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.openai_model = openai_model
        self.embedding_model = embedding_model
        self.tokenizer_openai = tiktoken.encoding_for_model(openai_model)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 1536  # text-embedding-ada-002 dimension
        
        # Teacher Model - Complex model với full episodic memory processing
        self.teacher_model = self._create_teacher_model()
        
        # Student Model - Lightweight model cho production
        self.student_model = self._create_student_model()
        
        # Optimizers
        self.teacher_optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=learning_rate)
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate * 2)
        
        # Loss functions
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.teacher_trained = False
    
    def _create_teacher_model(self) -> nn.Module:
        """Tạo Teacher Model - Mô hình phức tạp với full processing capability"""
        return nn.Sequential(
            # Complex feature extraction
            nn.Linear(self.embedding_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            # Memory consolidation layers
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Distilled knowledge representation
            nn.Softmax(dim=1)   # Soft probability distribution
        ).to(self.device)
        
    def _create_student_model(self) -> nn.Module:
        """Tạo Student Model - Mô hình lightweight cho production"""
        return nn.Sequential(
            # Simplified architecture
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),   # Same output dim as teacher
            nn.Softmax(dim=1)     # Soft probability distribution
        ).to(self.device)
    
    def distill_from_memories(self, 
                            memories: List[Dict[str, Any]], 
                            num_epochs: int = 5,
                            batch_size: int = 8) -> Dict[str, Any]:
        """
        Knowledge Distillation chính thức từ episodic memories
        Bao gồm:
        1. Train Teacher model trên memories 
        2. Distill knowledge sang Student model
        3. Sử dụng soft targets với temperature scaling
        """
        print(f"🎓 Bắt đầu Knowledge Distillation với {len(memories)} memories...")
        
        # Phase 1: Train Teacher Model
        teacher_results = self._train_teacher_model(memories, num_epochs, batch_size)
        
        if not teacher_results["success"]:
            return {
                "status": "failed",
                "reason": "teacher_training_failed",
                "teacher_results": teacher_results
            }
        
        # Phase 2: Distill to Student Model  
        distillation_results = self._distill_to_student(memories, num_epochs, batch_size)
        
        return {
            "status": "completed",
            "method": "knowledge_distillation",
            "teacher_training": teacher_results,
            "distillation": distillation_results,
            "parameters": {
                "temperature": self.temperature,
                "alpha": self.alpha,
                "beta": self.beta
            }
        }
    
    def _train_teacher_model(self, 
                           memories: List[Dict[str, Any]], 
                           num_epochs: int = 5,
                           batch_size: int = 8) -> Dict[str, Any]:
        """Train Teacher Model trên episodic memories"""
        print("👨‍🏫 Training Teacher Model...")
        
        # Prepare training data
        training_data = self._prepare_training_data(memories)
        if not training_data:
            return {"success": False, "reason": "no_valid_data"}
        
        self.teacher_model.train()
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # Get embeddings và rewards
                embeddings, rewards = self._prepare_teacher_batch(batch)
                if embeddings is None:
                    continue
                
                # Forward pass
                self.teacher_optimizer.zero_grad()
                
                embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.device)
                
                # Teacher prediction  
                teacher_output = self.teacher_model(embeddings_tensor)
                
                # Self-supervised loss (predict reward distribution)
                reward_targets = self._create_reward_targets(rewards_tensor)
                loss = self.cross_entropy_loss(teacher_output, reward_targets)
                
                # Backward pass
                loss.backward()
                self.teacher_optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                epoch_batches += 1
                total_batches += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            print(f"Teacher Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        self.teacher_trained = True
        return {
            "success": True,
            "avg_loss": total_loss / max(total_batches, 1),
            "epochs": num_epochs,
            "batches": total_batches
        }
    
    def _distill_to_student(self, 
                          memories: List[Dict[str, Any]], 
                          num_epochs: int = 5,
                          batch_size: int = 8) -> Dict[str, Any]:
        """Distill knowledge từ Teacher sang Student với soft targets"""
        print("🎒 Distilling knowledge to Student Model...")
        
        if not self.teacher_trained:
            return {"success": False, "reason": "teacher_not_trained"}
        
        training_data = self._prepare_training_data(memories)
        if not training_data:
            return {"success": False, "reason": "no_valid_data"}
        
        self.teacher_model.eval()  # Teacher ở mode evaluation
        self.student_model.train()  # Student ở mode training
        
        total_distillation_loss = 0.0
        total_student_loss = 0.0
        total_batches = 0
        
        for epoch in range(num_epochs):
            epoch_distillation_loss = 0.0
            epoch_student_loss = 0.0
            epoch_batches = 0
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                embeddings, rewards = self._prepare_teacher_batch(batch)
                if embeddings is None:
                    continue
                
                self.student_optimizer.zero_grad()
                
                embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.device)
                
                # Teacher soft targets với temperature scaling
                with torch.no_grad():
                    teacher_logits = self.teacher_model(embeddings_tensor)
                    teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
                
                # Student predictions
                student_logits = self.student_model(embeddings_tensor)
                student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
                
                # Distillation Loss (KL Divergence)
                distillation_loss = self.kl_div_loss(student_soft, teacher_soft) * (self.temperature ** 2)
                
                # Student Loss (original task)
                reward_targets = self._create_reward_targets(rewards_tensor)
                student_loss = self.cross_entropy_loss(student_logits, reward_targets)
                
                # Combined Loss
                total_loss = self.alpha * distillation_loss + self.beta * student_loss
                
                # Backward pass
                total_loss.backward()
                self.student_optimizer.step()
                
                epoch_distillation_loss += distillation_loss.item()
                epoch_student_loss += student_loss.item()
                total_distillation_loss += distillation_loss.item()
                total_student_loss += student_loss.item()
                epoch_batches += 1
                total_batches += 1
            
            avg_distill_loss = epoch_distillation_loss / max(epoch_batches, 1)
            avg_student_loss = epoch_student_loss / max(epoch_batches, 1)
            print(f"Student Epoch {epoch + 1}/{num_epochs}, Distill Loss: {avg_distill_loss:.4f}, Student Loss: {avg_student_loss:.4f}")
        
        return {
            "success": True,
            "avg_distillation_loss": total_distillation_loss / max(total_batches, 1),
            "avg_student_loss": total_student_loss / max(total_batches, 1),
            "epochs": num_epochs,
            "batches": total_batches
        }
    
    def _create_reward_targets(self, rewards: torch.Tensor) -> torch.Tensor:
        """Tạo categorical targets từ reward values"""
        # Normalize rewards to [0, 1] và convert to categories
        normalized_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        # Convert to 64 categories (matching output dim)
        categories = (normalized_rewards * 63).long().clamp(0, 63)
        return categories
    
    def _prepare_training_data(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuẩn bị training data từ memories"""
        training_data = []
        
        for memory in memories:
            context = memory.get('context', '')
            content = memory.get('content', '')
            reward = memory.get('reward', 0.0)
            
            # Filter memories với content đầy đủ và reward hợp lệ
            if context and content and -1.0 <= reward <= 1.0:
                training_data.append({
                    'context': context,
                    'content': content,
                    'reward': reward,
                    'memory_id': memory.get('id', 'unknown')
                })
        
        print(f"📊 Prepared {len(training_data)} training samples từ {len(memories)} total memories")
        return training_data
    
    def _prepare_teacher_batch(self, 
                             batch: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Chuẩn bị batch cho teacher training"""
        embeddings = []
        rewards = []
        
        for item in batch:
            # Combine context và content cho comprehensive embedding
            combined_text = f"Context: {item['context']}\nContent: {item['content']}"
            embedding = self.get_openai_embedding(combined_text)
            
            if embedding.size > 0:
                embeddings.append(embedding)
                rewards.append(item['reward'])
        
        if not embeddings:
            return None, None
        
        return np.array(embeddings), np.array(rewards)
    
    def get_student_prediction(self, text: str) -> Dict[str, Any]:
        """Sử dụng Student Model đã được distill để predict"""
        if not self.teacher_trained:
            return {"error": "Model chưa được train"}
        
        # Get embedding cho text
        embedding = self.get_openai_embedding(text)
        if embedding.size == 0:
            return {"error": "Could not get embedding"}
        
        # Predict với Student Model
        self.student_model.eval()
        with torch.no_grad():
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            student_output = self.student_model(embedding_tensor)
            
            # Get confidence score từ distribution
            confidence = torch.max(student_output).item()
            prediction_class = torch.argmax(student_output).item()
            
        return {
            "prediction_class": prediction_class,
            "confidence": confidence,
            "distribution": student_output.cpu().numpy().tolist(),
            "model_type": "student",
            "compressed_size": f"{sum(p.numel() for p in self.student_model.parameters())} parameters"
        }
    
    def compare_teacher_student(self, text: str) -> Dict[str, Any]:
        """So sánh predictions giữa Teacher và Student models"""
        if not self.teacher_trained:
            return {"error": "Models chưa được train"}
        
        embedding = self.get_openai_embedding(text)
        if embedding.size == 0:
            return {"error": "Could not get embedding"}
        
        self.teacher_model.eval()
        self.student_model.eval()
        
        with torch.no_grad():
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            
            # Teacher prediction
            teacher_output = self.teacher_model(embedding_tensor)
            teacher_confidence = torch.max(teacher_output).item()
            teacher_class = torch.argmax(teacher_output).item()
            
            # Student prediction  
            student_output = self.student_model(embedding_tensor)
            student_confidence = torch.max(student_output).item()
            student_class = torch.argmax(student_output).item()
            
            # KL Divergence between predictions
            kl_div = F.kl_div(
                F.log_softmax(student_output, dim=1),
                F.softmax(teacher_output, dim=1),
                reduction='batchmean'
            ).item()
        
        return {
            "teacher": {
                "prediction_class": teacher_class,
                "confidence": teacher_confidence,
                "model_size": f"{sum(p.numel() for p in self.teacher_model.parameters())} parameters"
            },
            "student": {
                "prediction_class": student_class,
                "confidence": student_confidence,
                "model_size": f"{sum(p.numel() for p in self.student_model.parameters())} parameters"
            },
            "similarity": {
                "prediction_match": teacher_class == student_class,
                "kl_divergence": kl_div,
                "confidence_diff": abs(teacher_confidence - student_confidence)
            },
            "compression_ratio": sum(p.numel() for p in self.teacher_model.parameters()) / sum(p.numel() for p in self.student_model.parameters())
        }
    
    def save_models(self, filepath_prefix: str) -> Dict[str, str]:
        """Lưu cả Teacher và Student models"""
        teacher_path = f"{filepath_prefix}_teacher.pth"
        student_path = f"{filepath_prefix}_student.pth"
        
        torch.save({
            'model_state_dict': self.teacher_model.state_dict(),
            'optimizer_state_dict': self.teacher_optimizer.state_dict(),
            'embedding_dim': self.embedding_dim,
            'trained': self.teacher_trained
        }, teacher_path)
        
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.student_optimizer.state_dict(),
            'embedding_dim': self.embedding_dim,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta
        }, student_path)
        
        return {
            "teacher_model": teacher_path,
            "student_model": student_path
        }
    
    def load_models(self, filepath_prefix: str) -> bool:
        """Load cả Teacher và Student models"""
        teacher_path = f"{filepath_prefix}_teacher.pth"
        student_path = f"{filepath_prefix}_student.pth"
        
        try:
            # Load Teacher
            teacher_checkpoint = torch.load(teacher_path, map_location=self.device)
            self.teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
            self.teacher_optimizer.load_state_dict(teacher_checkpoint['optimizer_state_dict'])
            self.teacher_trained = teacher_checkpoint.get('trained', False)
            
            # Load Student
            student_checkpoint = torch.load(student_path, map_location=self.device)
            self.student_model.load_state_dict(student_checkpoint['model_state_dict'])
            self.student_optimizer.load_state_dict(student_checkpoint['optimizer_state_dict'])
            
            return True
        except Exception as e:
            print(f"Lỗi khi load models: {e}")
            return False
    
    def _distill_with_openai(self, 
                           memories: List[Dict[str, Any]], 
                           num_epochs: int = 3,
                           batch_size: int = 4) -> Dict[str, Any]:
        """DEPRECATED: Distill knowledge sử dụng OpenAI embeddings - Use distill_from_memories instead"""
        print(f"🔄 Bắt đầu OpenAI distillation với {len(memories)} memories...")
        
        # Prepare data với OpenAI embeddings
        distillation_data = self._prepare_openai_distillation_data(memories)
        
        if not distillation_data:
            return {
                "status": "failed",
                "reason": "no_valid_data",
                "memories_processed": 0,
                "method": "openai"
            }
        
        total_loss = 0.0
        total_batches = 0
        token_stats = {"total_tokens": 0, "avg_efficiency": 0.0}
        
        self.distillation_network.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            # Process in batches
            for i in range(0, len(distillation_data), batch_size):
                batch = distillation_data[i:i + batch_size]
                
                # Prepare batch embeddings
                batch_embeddings, batch_targets = self._prepare_embedding_batch(batch)
                
                if batch_embeddings is None:
                    continue
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Convert embeddings to torch tensors
                embeddings_tensor = torch.FloatTensor(batch_embeddings).to(self.device)
                targets_tensor = torch.FloatTensor(batch_targets).to(self.device)
                
                # Distillation network forward
                distilled_features = self.distillation_network(embeddings_tensor)
                
                # Reconstruction loss
                loss = nn.MSELoss()(distilled_features, targets_tensor[:, :128])  # Match output dim
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                epoch_batches += 1
                total_batches += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Calculate final metrics
        avg_loss = total_loss / max(total_batches, 1)
        
        # Token efficiency analysis
        for data in distillation_data:
            efficiency = self._calculate_token_efficiency(data["context"])
            token_stats["total_tokens"] += efficiency["token_count"]
            token_stats["avg_efficiency"] += efficiency["efficiency_score"]
        
        token_stats["avg_efficiency"] /= max(len(distillation_data), 1)
        
        return {
            "status": "completed",
            "method": "openai",
            "avg_loss": avg_loss,
            "total_batches": total_batches,
            "epochs": num_epochs,
            "memories_processed": len(distillation_data),
            "token_statistics": token_stats,
            "embedding_cache_size": len(self.embedding_cache)
        }
    

    
    def get_openai_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding từ OpenAI API với caching"""
        # Check cache trước
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # Call OpenAI embedding API
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = np.array(response.data[0].embedding)
            
            # Cache kết quả
            self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"Lỗi khi lấy embedding: {e}")
            # Fallback: random embedding
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _calculate_token_efficiency(self, text: str) -> Dict[str, Any]:
        """Tính toán hiệu quả sử dụng token với OpenAI tokenizer"""
        try:
            tokens = self.tokenizer_openai.encode(text)
            return {
                "token_count": len(tokens),
                "character_count": len(text),
                "tokens_per_char": len(tokens) / max(len(text), 1),
                "efficiency_score": len(text) / max(len(tokens), 1)  # chars per token
            }
        except Exception as e:
            print(f"Lỗi khi tokenize: {e}")
            return {"token_count": 0, "efficiency_score": 0}
    
    def _prepare_openai_distillation_data(self, 
                                        memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuẩn bị data cho distillation với OpenAI embeddings"""
        distillation_data = []
        
        for memory in memories:
            context = memory.get('context', '')
            content = memory.get('content', '')
            reward = memory.get('reward', 0.0)
            
            # Chỉ sử dụng memories với reward tốt
            if context and content and reward > 0.3:
                distillation_data.append({
                    'context': context,
                    'content': content,
                    'reward': reward,
                    'memory_id': memory.get('id', 'unknown')
                })
        
        print(f"📊 Filtered {len(distillation_data)} good memories từ {len(memories)} total")
        return distillation_data
    
    def _prepare_embedding_batch(self, 
                               batch: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Chuẩn bị batch embeddings cho OpenAI distillation"""
        context_embeddings = []
        content_embeddings = []
        
        for item in batch:
            # Get embeddings cho context và content
            context_emb = self.get_openai_embedding(item['context'])
            content_emb = self.get_openai_embedding(item['content'])
            
            if context_emb.size > 0 and content_emb.size > 0:
                context_embeddings.append(context_emb)
                content_embeddings.append(content_emb)
        
        if not context_embeddings:
            return None, None
        
        return np.array(context_embeddings), np.array(content_embeddings)
    
    def extract_distilled_knowledge(self, text: str) -> Dict[str, Any]:
        """Extract distilled knowledge representation từ text"""
        return self._extract_with_openai(text)
    
    def _extract_with_openai(self, text: str) -> Dict[str, Any]:
        """Extract knowledge sử dụng OpenAI embeddings"""
        # Get OpenAI embedding
        embedding = self.get_openai_embedding(text)
        
        if embedding.size == 0:
            return {"error": "Could not get embedding"}
        
        # Process qua distillation network
        with torch.no_grad():
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            distilled_features = self.distillation_network(embedding_tensor)
            
        # Token analysis
        token_info = self._calculate_token_efficiency(text)
        
        return {
            "distilled_features": distilled_features.cpu().numpy().tolist(),
            "original_embedding_dim": len(embedding),
            "distilled_dim": distilled_features.shape[1],
            "compression_ratio": len(embedding) / distilled_features.shape[1],
            "token_analysis": token_info,
            "method": "openai"
        }
    



class MemoryConsolidationSystem:
    """Main system cho Memory Consolidation sử dụng OpenAI"""
    
    def __init__(self,
                 consolidation_threshold: int = 50,  # Số memories để trigger consolidation
                 consolidation_interval_hours: int = 24,
                 openai_model: str = "gpt-4o-mini",
                 api_key: str = None):

        self.consolidation_threshold = consolidation_threshold
        self.consolidation_interval = timedelta(hours=consolidation_interval_hours)
        self.last_consolidation = datetime.now()
        self.openai_model = openai_model
        
        # Initialize components
        self.summarizer = LLMSummarizer(openai_model=openai_model, api_key=api_key)
        self.knowledge_graph = KnowledgeGraph()
        self.distillation = ModelDistillation(
            openai_model=openai_model,
            api_key=api_key
        )
        
        # Storage for consolidated knowledge
        self.consolidated_knowledge: Dict[str, ConsolidatedKnowledge] = {}
        
        # OpenAI client for enhanced summarization
        try:
            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            print(f"✅ OpenAI integration enabled với model: {openai_model}")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi khởi tạo OpenAI client: {e}")
    
    def should_consolidate(self, num_new_memories: int) -> bool:
        """Kiểm tra xem có nên chạy consolidation không"""
        time_condition = datetime.now() - self.last_consolidation >= self.consolidation_interval
        memory_condition = num_new_memories >= self.consolidation_threshold
        
        return time_condition or memory_condition
    
    def consolidate_memories(self, 
                           episodic_memories: List[Dict[str, Any]],
                           method: str = "all",
                           auto_save: bool = True,
                           save_filepath: str = None) -> Dict[str, Any]:
        """Chạy memory consolidation với các method khác nhau"""
        results = {
            "summarization": None,
            "graph_integration": None,
            "distillation": None,
            "total_memories_processed": len(episodic_memories)
        }
        
        if method in ["summarization", "all"]:
            results["summarization"] = self._consolidate_via_summarization(episodic_memories)
        
        if method in ["graph", "all"]:
            results["graph_integration"] = self._consolidate_via_graph(episodic_memories)
        
        if method in ["distillation", "all"]:
            results["distillation"] = self._consolidate_via_distillation(episodic_memories)
        
        self.last_consolidation = datetime.now()
        
        # Tự động lưu dữ liệu nếu được yêu cầu
        if auto_save and save_filepath:
            try:
                self.save_consolidated_knowledge(save_filepath)
                results["saved_to_file"] = save_filepath
                print(f"💾 Đã lưu consolidated knowledge vào: {save_filepath}")
            except Exception as e:
                print(f"⚠️  Lỗi khi lưu consolidated knowledge: {e}")
                results["save_error"] = str(e)
        
        return results
    
    def enhance_summarization_with_openai(self, memories: List[Dict[str, Any]]) -> str:
        """Enhanced summarization sử dụng OpenAI API"""
        try:
            # Prepare context từ memories
            memory_texts = []
            for memory in memories:
                text = f"Context: {memory.get('context', '')}\nContent: {memory.get('content', '')}\nReward: {memory.get('reward', 0)}"
                memory_texts.append(text)
            
            combined_text = "\n---\n".join(memory_texts)
            
            # Create enhanced prompt
            prompt = f"""Phân tích và tóm tắt các trải nghiệm hội thoại sau thành kiến thức hữu ích và có cấu trúc:

{combined_text}

Hãy tạo một tóm tắt ngắn gọn (2-3 câu) về:
1. Chủ đề chính
2. Patterns hành vi người dùng
3. Kiến thức quan trọng cần ghi nhớ

Tóm tắt:"""

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "Bạn là một AI chuyên phân tích và tóm tắt dữ liệu cuộc hội thoại."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️  Lỗi OpenAI summarization, fallback to basic: {e}")
            return self.summarizer.summarize_memories(memories)
    
    def _consolidate_via_summarization(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidation qua summarization"""
        # Group memories by similarity or topic (simplified)
        memory_groups = self._group_memories_by_similarity(memories)
        
        consolidated_count = 0
        summaries_created = []
        
        for group in memory_groups:
            if len(group) >= 3:                  # Chỉ consolidate nếu có ít nhất 3 memories
                # Sử dụng enhanced summarization với OpenAI
                summary = self.enhance_summarization_with_openai(group)
                concepts = self.summarizer.extract_key_concepts(summary)
                
                # Create consolidated knowledge
                knowledge_id = f"summary_{datetime.now().timestamp()}"
                consolidated = ConsolidatedKnowledge(
                    id=knowledge_id,
                    summary=summary,
                    source_memories=[m.get('id', '') for m in group],
                    confidence_score=len(group) / len(memories),  # Simple confidence
                    consolidation_method="summarization",
                    created_at=datetime.now()
                )
                
                self.consolidated_knowledge[knowledge_id] = consolidated
                summaries_created.append({
                    "id": knowledge_id,
                    "summary": summary,
                    "concepts": concepts,
                    "source_count": len(group)
                })
                consolidated_count += len(group)
        
        return {
            "method": "summarization",
            "memories_consolidated": consolidated_count,
            "summaries_created": len(summaries_created),
            "summaries": summaries_created
        }
    
    @profile_function
    def _consolidate_via_graph(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidation qua knowledge graph integration - Tối ưu hóa hiệu suất"""
        if not memories:
            return {
                "method": "graph_integration",
                "concepts_added": 0,
                "edges_added": 0,
                "graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
                "graph_edges": self.knowledge_graph.graph.number_of_edges()
            }
        
        # Pre-allocate containers để tránh dynamic allocation
        consolidated_items = []
        current_time = datetime.now()  # Cache timestamp
        
        # Batch process memories để tối ưu hóa
        for memory in memories:
            content = memory.get('content', '')
            if not content:  # Skip empty content
                continue
                
            concepts = self.summarizer.extract_key_concepts(content)
            if not concepts:
                continue
            
            # Create consolidated knowledge
            knowledge_id = f"graph_{memory.get('id', current_time.timestamp())}"
            summary = content[:200] + "..." if len(content) > 200 else content
            
            consolidated = ConsolidatedKnowledge(
                id=knowledge_id,
                summary=summary,
                source_memories=[memory.get('id', '')],
                confidence_score=1.0,
                consolidation_method="graph",
                created_at=current_time
            )
            
            consolidated_items.append((consolidated, concepts))
        
        # Batch add to knowledge graph (tối ưu hóa networkx operations)
        concepts_added = 0
        edges_added = 0
        
        for consolidated, concepts in consolidated_items:
            self.knowledge_graph.add_knowledge(consolidated, concepts)
            self.consolidated_knowledge[consolidated.id] = consolidated
            
            concepts_added += len(concepts)
            edges_added += len(concepts)
        
        return {
            "method": "graph_integration",
            "concepts_added": concepts_added,
            "edges_added": edges_added,
            "graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
            "graph_edges": self.knowledge_graph.graph.number_of_edges()
        }
    
    def _consolidate_via_distillation(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidation qua model distillation"""
        # Filter memories with good rewards for distillation
        good_memories = [m for m in memories if m.get('reward', 0) > 0.5]
        
        if len(good_memories) < 10:  # Need minimum memories for distillation
            return {
                "method": "distillation",
                "status": "skipped",
                "reason": "insufficient_good_memories",
                "available_memories": len(good_memories)
            }
        
        # Run distillation
        distillation_results = self.distillation.distill_from_memories(good_memories)
        
        return {
            "method": "distillation",
            "status": "completed",
            "memories_used": len(good_memories),
            **distillation_results
        }
    
    @profile_function
    def _group_memories_by_similarity(self, 
                                    memories: List[Dict[str, Any]], 
                                    similarity_threshold: float = 0.7) -> List[List[Dict[str, Any]]]:
        """Group memories theo similarity - Tối ưu hóa hiệu suất với vectorization"""
        if not memories:
            return []
        
        # Pre-compute word sets để tránh repeated computation
        memory_word_sets = []
        valid_memories = []
        
        for memory in memories:
            content = memory.get('content', '')
            if content:
                words = set(content.lower().split())
                if words:  # Chỉ xử lý memories có content
                    memory_word_sets.append(words)
                    valid_memories.append(memory)
        
        if not valid_memories:
            return []
        
        # Pre-allocate containers
        groups = []
        used_indices = set()
        num_memories = len(valid_memories)
        
        # Vectorized similarity computation
        for i in range(num_memories):
            if i in used_indices:
                continue
            
            current_words = memory_word_sets[i]
            group = [valid_memories[i]]
            used_indices.add(i)
            
            # Batch similarity check với vectorized operations
            for j in range(i + 1, num_memories):
                if j in used_indices:
                    continue
                
                other_words = memory_word_sets[j]
                
                # Optimized similarity calculation
                if current_words and other_words:
                    intersection_size = len(current_words.intersection(other_words))
                    if intersection_size > 0:  # Early exit nếu không có overlap
                        union_size = len(current_words.union(other_words))
                        similarity = intersection_size / union_size
                        
                        if similarity >= similarity_threshold:
                            group.append(valid_memories[j])
                            used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def query_consolidated_knowledge(self, 
                                   query: str, 
                                   method: str = "all") -> List[Dict[str, Any]]:
        """Query consolidated knowledge"""
        results = []
        
        if method in ["summarization", "all"]:
            # Search in summaries
            for knowledge in self.consolidated_knowledge.values():
                if knowledge.consolidation_method == "summarization":
                    if query.lower() in knowledge.summary.lower():
                        results.append({
                            "type": "summary",
                            "content": knowledge.summary,
                            "confidence": knowledge.confidence_score,
                            "source_memories": len(knowledge.source_memories)
                        })
        
        if method in ["graph", "all"]:
            # Search in knowledge graph
            query_concepts = self.summarizer.extract_key_concepts(query)
            graph_results = self.knowledge_graph.find_related_knowledge(query_concepts)
            
            for result in graph_results:
                results.append({
                    "type": "graph",
                    "content": result['summary'],
                    "confidence": result['confidence'],
                    "matching_concept": result['matching_concept']
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:10]  # Return top 10
    
    def save_consolidated_knowledge(self, filepath: str) -> None:
        """Lưu consolidated knowledge"""
        # Tạo thư mục nếu cần thiết
        directory = os.path.dirname(filepath)
        if directory:  # Chỉ tạo thư mục nếu có đường dẫn
            os.makedirs(directory, exist_ok=True)
        
        data = {
            "consolidated_knowledge": {
                k: v.to_dict() for k, v in self.consolidated_knowledge.items()
            },
            "last_consolidation": self.last_consolidation.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save knowledge graph separately
        graph_filepath = filepath.replace('.json', '_graph.pkl')
        self.knowledge_graph.save_graph(graph_filepath)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Lấy thông tin về system và OpenAI integration"""
        info = {
            "consolidation_threshold": self.consolidation_threshold,
            "consolidation_interval_hours": self.consolidation_interval.total_seconds() / 3600,
            "last_consolidation": self.last_consolidation.isoformat(),
            "total_consolidated_knowledge": len(self.consolidated_knowledge),
            "knowledge_graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
            "knowledge_graph_edges": self.knowledge_graph.graph.number_of_edges(),
            "openai_integration": {
                "enabled": True,
                "available": True,
                "model": self.openai_model
            }
        }
        
        info["openai_integration"].update({
            "embedding_model": getattr(self.distillation, 'embedding_model', 'unknown'),
            "embedding_cache_size": len(getattr(self.distillation, 'embedding_cache', {})),
            "distillation_method": "openai_embeddings"
        })
        
        return info
    
    def clear_embedding_cache(self) -> int:
        """Xóa embedding cache để tiết kiệm memory"""
        if hasattr(self.distillation, 'embedding_cache'):
            cache_size = len(self.distillation.embedding_cache)
            self.distillation.embedding_cache.clear()
            return cache_size
        return 0
    
    def force_save_knowledge(self, filepath: str = None) -> Dict[str, Any]:
        """Bắt buộc lưu tất cả knowledge hiện tại"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"consolidated_knowledge_{timestamp}.json"
        
        try:
            self.save_consolidated_knowledge(filepath)
            return {
                "status": "success",
                "filepath": filepath,
                "knowledge_count": len(self.consolidated_knowledge),
                "graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
                "graph_edges": self.knowledge_graph.graph.number_of_edges()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "filepath": filepath
            }
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Lấy trạng thái lưu trữ hiện tại"""
        return {
            "memory_storage": {
                "consolidated_knowledge_count": len(self.consolidated_knowledge),
                "knowledge_graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
                "knowledge_graph_edges": self.knowledge_graph.graph.number_of_edges(),
                "concept_memories_count": len(self.knowledge_graph.concept_memories)
            },
            "last_consolidation": self.last_consolidation.isoformat(),
            "consolidation_threshold": self.consolidation_threshold,
            "next_consolidation_due": (self.last_consolidation + self.consolidation_interval).isoformat()
        }
