"""
Retrieval-Augmented Episodic Memory System
Sử dụng vector search để tìm kiếm và truy xuất memories
"""

import os
import json
import numpy as np
import faiss
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import pickle


@dataclass
class EpisodicMemory:
    """Đại diện cho một memory trong episodic memory system"""
    id: str
    content: str  # Nội dung memory
    context: str  # Context xung quanh memory
    embedding: Optional[List[float]] = None
    timestamp: datetime = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance_score: float = 1.0
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
        """Convert memory thành dictionary"""
        data = asdict(self)
        # Convert datetime objects to strings
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Tạo EpisodicMemory từ dictionary"""
        # Convert timestamp strings back to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'last_accessed' in data and isinstance(data['last_accessed'], str):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class VectorMemoryStore:
    """Vector store sử dụng FAISS để lưu trữ và search memories"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 index_type: str = "flat"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.memories: Dict[str, EpisodicMemory] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
    
    def add_memory(self, memory: EpisodicMemory) -> None:
        """Thêm memory vào vector store"""
        # Generate embedding if not exists
        if memory.embedding is None:
            text_to_embed = f"{memory.content} {memory.context}"
            embedding = self.embedding_model.encode([text_to_embed])[0]
            memory.embedding = embedding.tolist()
        
        # Add to FAISS index
        embedding_array = np.array([memory.embedding], dtype=np.float32)
        self.index.add(embedding_array)
        
        # Update mappings
        index_position = self.index.ntotal - 1
        self.id_to_index[memory.id] = index_position
        self.index_to_id[index_position] = memory.id
        self.memories[memory.id] = memory
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = 5,
                      threshold: float = 0.7) -> List[Tuple[EpisodicMemory, float]]:
        """Tìm kiếm memories tương tự với query"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Convert distance to similarity score
            similarity = 1 / (1 + distance)
            
            if similarity >= threshold:
                memory_id = self.index_to_id[idx]
                memory = self.memories[memory_id]
                results.append((memory, similarity))
        
        return results
    
    def get_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Lấy memory theo ID"""
        return self.memories.get(memory_id)
    
    def update_memory_access(self, memory_id: str) -> None:
        """Cập nhật thông tin access cho memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
    
    def remove_memory(self, memory_id: str) -> bool:
        """Xóa memory khỏi store"""
        if memory_id not in self.memories:
            return False
        
        # Note: FAISS doesn't support direct removal
        # Trong thực tế, cần rebuild index hoặc mark as deleted
        del self.memories[memory_id]
        if memory_id in self.id_to_index:
            del self.id_to_index[memory_id]
        
        return True


class ChromaMemoryStore:
    """Memory store sử dụng ChromaDB"""
    
    def __init__(self, 
                 collection_name: str = "episodic_memories",
                 persist_directory: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Episodic memories for chatbot"}
        )
        self.memories: Dict[str, EpisodicMemory] = {}
    
    def add_memory(self, memory: EpisodicMemory) -> None:
        """Thêm memory vào ChromaDB"""
        text_to_embed = f"{memory.content} {memory.context}"
        
        self.collection.add(
            documents=[text_to_embed],
            metadatas=[{
                "timestamp": memory.timestamp.isoformat(),
                "importance_score": memory.importance_score,
                "tags": ",".join(memory.tags),
                "access_count": memory.access_count
            }],
            ids=[memory.id]
        )
        
        self.memories[memory.id] = memory
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = 5,
                      where_filter: Optional[Dict] = None) -> List[Tuple[EpisodicMemory, float]]:
        """Tìm kiếm memories tương tự"""
        if not self.memories:  # No memories to search
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.memories)),  # Don't exceed available memories
            where=where_filter
        )
        
        memories_with_scores = []
        if results['ids'][0]:  # Check if results exist
            for memory_id, distance in zip(results['ids'][0], results['distances'][0]):
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    # Convert distance to similarity (ChromaDB returns cosine distance)
                    # Clamp distance to reasonable range và normalize better
                    distance = max(0.0, min(2.0, distance))  # Cosine distance range [0, 2]
                    similarity = 1 - (distance / 2.0)  # Normalize to [0, 1]
                    similarity = max(0.0, min(1.0, similarity))  # Ensure [0, 1] range
                    memories_with_scores.append((memory, similarity))
        
        # Sort by similarity descending
        memories_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return memories_with_scores
    
    def get_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Lấy memory theo ID"""
        return self.memories.get(memory_id)
    
    def update_memory_access(self, memory_id: str) -> None:
        """Cập nhật access information"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            
            # Update in ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[{
                    "timestamp": memory.timestamp.isoformat(),
                    "importance_score": memory.importance_score,
                    "tags": ",".join(memory.tags),
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed.isoformat()
                }]
            )


class RetrievalAugmentedMemory:
    """Main class cho Retrieval-Augmented Episodic Memory system"""
    
    def __init__(self,
                 store_type: str = "chroma",  # "faiss" or "chroma"
                 max_memories: int = 10000,
                 decay_factor: float = 0.95,
                 importance_threshold: float = 0.1):
        self.max_memories = max_memories
        self.decay_factor = decay_factor
        self.importance_threshold = importance_threshold
        
        # Initialize memory store
        if store_type == "faiss":
            self.store = VectorMemoryStore()
        else:
            self.store = ChromaMemoryStore()
        
        self.forgetting_scheduler_days = 7  # Run forgetting every 7 days
        self.last_forgetting_run = datetime.now()
    
    def add_memory(self, 
                  content: str, 
                  context: str = "",
                  tags: List[str] = None,
                  importance_score: float = 1.0,
                  metadata: Dict[str, Any] = None) -> str:
        """Thêm memory mới"""
        memory_id = str(uuid.uuid4())
        memory = EpisodicMemory(
            id=memory_id,
            content=content,
            context=context,
            tags=tags or [],
            importance_score=importance_score,
            metadata=metadata or {}
        )
        
        self.store.add_memory(memory)
        
        # Check if need to run forgetting mechanism
        self._check_and_run_forgetting()
        
        return memory_id
    
    def retrieve_relevant_memories(self, 
                                 query: str,
                                 top_k: int = 5,
                                 context_window: int = 3) -> List[Dict[str, Any]]:
        """Truy xuất memories liên quan đến query"""
        similar_memories = self.store.search_similar(query, top_k)
        
        retrieved_memories = []
        for memory, similarity in similar_memories:
            # Update access information
            self.store.update_memory_access(memory.id)
            
            # Prepare memory data for return
            memory_data = {
                "id": memory.id,
                "content": memory.content,
                "context": memory.context,
                "similarity": similarity,
                "importance_score": memory.importance_score,
                "access_count": memory.access_count,
                "timestamp": memory.timestamp,
                "tags": memory.tags,
                "metadata": memory.metadata
            }
            retrieved_memories.append(memory_data)
        
        return retrieved_memories
    
    def update_memory_importance(self, 
                               memory_id: str, 
                               importance_delta: float) -> bool:
        """Cập nhật importance score của memory"""
        memory = self.store.get_memory(memory_id)
        if memory:
            memory.importance_score = max(0.0, memory.importance_score + importance_delta)
            return True
        return False
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Thống kê về memory system"""
        total_memories = len(self.store.memories)
        if total_memories == 0:
            return {"total_memories": 0}
        
        importance_scores = [mem.importance_score for mem in self.store.memories.values()]
        access_counts = [mem.access_count for mem in self.store.memories.values()]
        
        return {
            "total_memories": total_memories,
            "avg_importance": np.mean(importance_scores),
            "max_importance": max(importance_scores),
            "min_importance": min(importance_scores),
            "avg_access_count": np.mean(access_counts),
            "max_access_count": max(access_counts),
            "highly_important_memories": sum(1 for score in importance_scores if score > 2.0),
            "low_importance_memories": sum(1 for score in importance_scores if score < 0.5)
        }
    
    def _check_and_run_forgetting(self) -> None:
        """Kiểm tra và chạy forgetting mechanism"""
        days_since_last_run = (datetime.now() - self.last_forgetting_run).days
        
        if days_since_last_run >= self.forgetting_scheduler_days:
            self._run_forgetting_mechanism()
            self.last_forgetting_run = datetime.now()
    
    def _run_forgetting_mechanism(self) -> int:
        """Chạy cơ chế forgetting để xóa memories ít quan trọng"""
        if len(self.store.memories) <= self.max_memories * 0.8:
            return 0  # Không cần forgetting
        
        # Calculate forgetting scores
        forgetting_candidates = []
        current_time = datetime.now()
        
        for memory in self.store.memories.values():
            # Temporal decay
            days_old = (current_time - memory.timestamp).days
            temporal_decay = self.decay_factor ** days_old
            
            # Access-based importance
            access_importance = np.log(1 + memory.access_count)
            
            # Combined forgetting score (lower = more likely to forget)
            forgetting_score = memory.importance_score * temporal_decay * access_importance
            
            if forgetting_score < self.importance_threshold:
                forgetting_candidates.append((memory.id, forgetting_score))
        
        # Sort by forgetting score (ascending)
        forgetting_candidates.sort(key=lambda x: x[1])
        
        # Remove lowest scoring memories
        memories_to_remove = len(self.store.memories) - int(self.max_memories * 0.8)
        removed_count = 0
        
        for memory_id, _ in forgetting_candidates[:memories_to_remove]:
            if self.store.remove_memory(memory_id):
                removed_count += 1
        
        return removed_count
    
    def save_memories(self, filepath: str) -> None:
        """Lưu tất cả memories ra file"""
        memories_data = []
        for memory in self.store.memories.values():
            memories_data.append(memory.to_dict())
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memories_data, f, ensure_ascii=False, indent=2)
    
    def load_memories(self, filepath: str) -> int:
        """Load memories từ file"""
        if not os.path.exists(filepath):
            return 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            memories_data = json.load(f)
        
        loaded_count = 0
        for memory_data in memories_data:
            try:
                memory = EpisodicMemory.from_dict(memory_data)
                self.store.add_memory(memory)
                loaded_count += 1
            except Exception as e:
                print(f"Lỗi khi load memory: {e}")
        
        return loaded_count
