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
    consolidation_method: str  # "summarization", "graph"
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


# ModelDistillation class removed - không được sử dụng hiệu quả trong hệ thống
# Thay thế bằng OpenAI API và các phương pháp consolidation đơn giản hơn
    

    

    

    


    

    

    

    

    



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
        # ModelDistillation removed - không hiệu quả
        
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
            "total_memories_processed": len(episodic_memories)
        }
        
        if method in ["summarization", "all"]:
            results["summarization"] = self._consolidate_via_summarization(episodic_memories)
        
        if method in ["graph", "all"]:
            results["graph_integration"] = self._consolidate_via_graph(episodic_memories)
        
        # Distillation method removed - không hiệu quả
        
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
    
    # _consolidate_via_distillation method removed - không hiệu quả
    
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
        
        # Distillation integration removed
        
        return info
    
    def clear_embedding_cache(self) -> int:
        """Xóa embedding cache để tiết kiệm memory - distillation removed"""
        # ModelDistillation đã bị xóa, không còn cache để clear
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
