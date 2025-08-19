"""
Episodic Memory Consolidation System
Chuyển đổi từ episodic memory sang semantic memory (weights) giống hippocampus
"""

import json
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import pickle
import os

# OpenAI imports
try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI không có sẵn. Chỉ sử dụng HuggingFace models.")


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
    """Sử dụng LLM để tóm tắt nhiều episodic memories thành knowledge chunks"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def summarize_memories(self, 
                          memories: List[Dict[str, Any]], 
                          max_length: int = 150) -> str:
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
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and extract summary
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_response[len(prompt):].strip()
        
        return summary
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Trích xuất các concept chính từ text"""
        # Simplified concept extraction (trong thực tế có thể dùng NER hoặc topic modeling)
        words = text.lower().split()
        
        # Filter out common words and extract potential concepts
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = []
        
        for i, word in enumerate(words):
            if len(word) > 3 and word not in stopwords:
                # Look for compound concepts (2-3 words)
                if i < len(words) - 1:
                    compound = f"{word} {words[i+1]}"
                    if len(compound) > 6:
                        concepts.append(compound)
                concepts.append(word)
        
        # Remove duplicates and return top concepts
        unique_concepts = list(set(concepts))
        return unique_concepts[:10]


class KnowledgeGraph:
    """Knowledge Graph để tích hợp consolidated knowledge"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings: Dict[str, List[float]] = {}
        self.concept_memories: Dict[str, List[str]] = defaultdict(list)
    
    def add_knowledge(self, 
                     consolidated_knowledge: ConsolidatedKnowledge,
                     concepts: List[str]) -> None:
        """Thêm consolidated knowledge vào graph"""
        knowledge_id = consolidated_knowledge.id
        
        # Add knowledge node
        self.graph.add_node(
            knowledge_id,
            type="knowledge",
            summary=consolidated_knowledge.summary,
            confidence=consolidated_knowledge.confidence_score,
            method=consolidated_knowledge.consolidation_method
        )
        
        # Add concept nodes and edges
        for concept in concepts:
            concept_id = f"concept_{concept.replace(' ', '_')}"
            
            # Add concept node if not exists
            if concept_id not in self.graph:
                self.graph.add_node(concept_id, type="concept", name=concept)
            
            # Add edge from knowledge to concept
            self.graph.add_edge(knowledge_id, concept_id, relation="contains")
            
            # Track which memories contributed to this concept
            self.concept_memories[concept].extend(consolidated_knowledge.source_memories)
    
    def find_related_knowledge(self, 
                             query_concepts: List[str], 
                             max_depth: int = 2) -> List[Dict[str, Any]]:
        """Tìm knowledge liên quan đến query concepts"""
        related_knowledge = []
        
        for concept in query_concepts:
            concept_id = f"concept_{concept.replace(' ', '_')}"
            
            if concept_id in self.graph:
                # Find knowledge nodes connected to this concept
                for knowledge_id in self.graph.predecessors(concept_id):
                    if self.graph.nodes[knowledge_id]['type'] == 'knowledge':
                        knowledge_data = {
                            'id': knowledge_id,
                            'summary': self.graph.nodes[knowledge_id]['summary'],
                            'confidence': self.graph.nodes[knowledge_id]['confidence'],
                            'method': self.graph.nodes[knowledge_id]['method'],
                            'matching_concept': concept
                        }
                        related_knowledge.append(knowledge_data)
        
        # Remove duplicates and sort by confidence
        unique_knowledge = {k['id']: k for k in related_knowledge}
        sorted_knowledge = sorted(
            unique_knowledge.values(), 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        return sorted_knowledge
    
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
    """Knowledge Distillation để fine-tune base model với episodic memories
    Hỗ trợ cả HuggingFace và OpenAI API"""
    
    def __init__(self, 
                 base_model_name: str = "microsoft/DialoGPT-small",
                 learning_rate: float = 1e-5,
                 use_openai: bool = False,
                 openai_model: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002",
                 api_key: str = None):
        
        self.use_openai = use_openai and OPENAI_AVAILABLE
        
        if self.use_openai:
            # OpenAI setup
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.openai_model = openai_model
            self.embedding_model = embedding_model
            self.tokenizer_openai = tiktoken.encoding_for_model(openai_model)
            
            # Neural components cho OpenAI embeddings
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_dim = 1536  # text-embedding-ada-002 dimension
            
            # Distillation network cho OpenAI embeddings
            self.distillation_network = nn.Sequential(
                nn.Linear(self.embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.Tanh()
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.distillation_network.parameters(), lr=learning_rate)
            self.embedding_cache: Dict[str, np.ndarray] = {}
            
        else:
            # HuggingFace setup (original)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def distill_from_memories(self, 
                            memories: List[Dict[str, Any]], 
                            num_epochs: int = 3,
                            batch_size: int = 4) -> Dict[str, Any]:
        """Distill knowledge từ episodic memories vào model weights"""
        
        if self.use_openai:
            return self._distill_with_openai(memories, num_epochs, batch_size)
        else:
            return self._distill_with_huggingface(memories, num_epochs, batch_size)
    
    def _distill_with_openai(self, 
                           memories: List[Dict[str, Any]], 
                           num_epochs: int = 3,
                           batch_size: int = 4) -> Dict[str, Any]:
        """Distill knowledge sử dụng OpenAI embeddings"""
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
    
    def _distill_with_huggingface(self, 
                                memories: List[Dict[str, Any]], 
                                num_epochs: int = 3,
                                batch_size: int = 4) -> Dict[str, Any]:
        """Distill knowledge sử dụng HuggingFace models (original method)"""
        training_data = self._prepare_training_data(memories)
        
        total_loss = 0.0
        num_batches = 0
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Process in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # Prepare batch
                inputs, targets = self._prepare_batch(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**inputs, labels=targets)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / (len(training_data) // batch_size):.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "status": "completed",
            "method": "huggingface",
            "avg_loss": avg_loss,
            "total_batches": num_batches,
            "epochs": num_epochs
        }
    
    def _prepare_training_data(self, memories: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Chuẩn bị training data từ memories"""
        training_data = []
        
        for memory in memories:
            # Create input-output pairs from memory
            context = memory.get('context', '')
            content = memory.get('content', '')
            
            if context and content:
                training_data.append({
                    'input': context,
                    'output': content
                })
        
        return training_data
    
    def _prepare_batch(self, batch: List[Dict[str, str]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Chuẩn bị batch cho training"""
        inputs = []
        targets = []
        
        for item in batch:
            input_text = item['input']
            output_text = item['output']
            full_text = f"{input_text} {self.tokenizer.eos_token} {output_text}"
            
            # Tokenize
            encoded = self.tokenizer(
                full_text,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            inputs.append(encoded['input_ids'].squeeze())
            targets.append(encoded['input_ids'].squeeze())
        
        # Stack tensors
        input_ids = torch.stack(inputs)
        attention_mask = torch.ones_like(input_ids)
        target_ids = torch.stack(targets)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }, target_ids
    
    def get_openai_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding từ OpenAI API với caching"""
        if not self.use_openai:
            return np.array([])
            
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
        if not self.use_openai:
            return {"token_count": 0, "efficiency_score": 0}
            
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
        if not self.use_openai:
            return []
            
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
        if not self.use_openai:
            return None, None
            
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
        if self.use_openai:
            return self._extract_with_openai(text)
        else:
            return self._extract_with_huggingface(text)
    
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
    
    def _extract_with_huggingface(self, text: str) -> Dict[str, Any]:
        """Extract knowledge sử dụng HuggingFace model"""
        # Simple representation using model embeddings
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        
        with torch.no_grad():
            if hasattr(self.model, 'get_input_embeddings'):
                embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
                features = embeddings.mean(dim=1)  # Simple averaging
            else:
                features = torch.randn(1, 768)  # Fallback
        
        return {
            "distilled_features": features.cpu().numpy().tolist(),
            "original_embedding_dim": features.shape[1],
            "distilled_dim": features.shape[1],
            "compression_ratio": 1.0,
            "method": "huggingface"
        }


class MemoryConsolidationSystem:
    """Main system cho Memory Consolidation với OpenAI support"""
    
    def __init__(self,
                 consolidation_threshold: int = 50,  # Số memories để trigger consolidation
                 consolidation_interval_hours: int = 24,
                 use_openai: bool = False,
                 openai_model: str = "gpt-3.5-turbo",
                 api_key: str = None):
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_interval = timedelta(hours=consolidation_interval_hours)
        self.last_consolidation = datetime.now()
        self.use_openai = use_openai and OPENAI_AVAILABLE
        
        # Initialize components
        self.summarizer = LLMSummarizer()
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize ModelDistillation với OpenAI support
        self.distillation = ModelDistillation(
            use_openai=self.use_openai,
            openai_model=openai_model,
            api_key=api_key
        )
        
        # Storage for consolidated knowledge
        self.consolidated_knowledge: Dict[str, ConsolidatedKnowledge] = {}
        
        # OpenAI client for enhanced summarization (optional)
        if self.use_openai:
            try:
                self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                self.openai_model = openai_model
                print(f"✅ OpenAI integration enabled với model: {openai_model}")
            except Exception as e:
                print(f"⚠️  Lỗi khi khởi tạo OpenAI client: {e}")
                self.use_openai = False
    
    def should_consolidate(self, num_new_memories: int) -> bool:
        """Kiểm tra xem có nên chạy consolidation không"""
        time_condition = datetime.now() - self.last_consolidation >= self.consolidation_interval
        memory_condition = num_new_memories >= self.consolidation_threshold
        
        return time_condition or memory_condition
    
    def consolidate_memories(self, 
                           episodic_memories: List[Dict[str, Any]],
                           method: str = "all") -> Dict[str, Any]:
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
        return results
    
    def enhance_summarization_with_openai(self, memories: List[Dict[str, Any]]) -> str:
        """Enhanced summarization sử dụng OpenAI API"""
        if not self.use_openai:
            return self.summarizer.summarize_memories(memories)
        
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
            print(f"⚠️  Lỗi OpenAI summarization, fallback to local: {e}")
            return self.summarizer.summarize_memories(memories)
    
    def _consolidate_via_summarization(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidation qua summarization"""
        # Group memories by similarity or topic (simplified)
        memory_groups = self._group_memories_by_similarity(memories)
        
        consolidated_count = 0
        summaries_created = []
        
        for group in memory_groups:
            if len(group) >= 3:  # Chỉ consolidate nếu có ít nhất 3 memories
                # Sử dụng enhanced summarization với OpenAI nếu có
                if self.use_openai:
                    summary = self.enhance_summarization_with_openai(group)
                else:
                    summary = self.summarizer.summarize_memories(group)
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
    
    def _consolidate_via_graph(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidation qua knowledge graph integration"""
        concepts_added = 0
        edges_added = 0
        
        for memory in memories:
            content = memory.get('content', '')
            concepts = self.summarizer.extract_key_concepts(content)
            
            if concepts:
                # Create consolidated knowledge for this memory
                knowledge_id = f"graph_{memory.get('id', datetime.now().timestamp())}"
                consolidated = ConsolidatedKnowledge(
                    id=knowledge_id,
                    summary=content[:200] + "..." if len(content) > 200 else content,
                    source_memories=[memory.get('id', '')],
                    confidence_score=1.0,
                    consolidation_method="graph",
                    created_at=datetime.now()
                )
                
                # Add to knowledge graph
                self.knowledge_graph.add_knowledge(consolidated, concepts)
                self.consolidated_knowledge[knowledge_id] = consolidated
                
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
    
    def _group_memories_by_similarity(self, 
                                    memories: List[Dict[str, Any]], 
                                    similarity_threshold: float = 0.7) -> List[List[Dict[str, Any]]]:
        """Group memories theo similarity (simplified implementation)"""
        # Simplified grouping - trong thực tế sẽ dùng embedding similarity
        groups = []
        used_indices = set()
        
        for i, memory in enumerate(memories):
            if i in used_indices:
                continue
            
            group = [memory]
            used_indices.add(i)
            
            # Find similar memories (simplified by checking common words)
            memory_words = set(memory.get('content', '').lower().split())
            
            for j, other_memory in enumerate(memories):
                if j <= i or j in used_indices:
                    continue
                
                other_words = set(other_memory.get('content', '').lower().split())
                
                # Simple similarity based on word overlap
                if memory_words and other_words:
                    overlap = len(memory_words.intersection(other_words))
                    similarity = overlap / len(memory_words.union(other_words))
                    
                    if similarity >= similarity_threshold:
                        group.append(other_memory)
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
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
                "enabled": self.use_openai,
                "available": OPENAI_AVAILABLE
            }
        }
        
        if self.use_openai:
            info["openai_integration"].update({
                "model": getattr(self, 'openai_model', 'unknown'),
                "embedding_model": getattr(self.distillation, 'embedding_model', 'unknown'),
                "embedding_cache_size": len(getattr(self.distillation, 'embedding_cache', {})),
                "distillation_method": "openai_embeddings"
            })
        else:
            info["openai_integration"]["distillation_method"] = "huggingface_local"
        
        return info
    
    def clear_embedding_cache(self) -> int:
        """Xóa embedding cache để tiết kiệm memory"""
        if self.use_openai and hasattr(self.distillation, 'embedding_cache'):
            cache_size = len(self.distillation.embedding_cache)
            self.distillation.embedding_cache.clear()
            return cache_size
        return 0
