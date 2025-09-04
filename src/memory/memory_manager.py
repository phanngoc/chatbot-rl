"""
Intelligent Memory Manager implementing Algorithm 1
Quyết định thông minh về các memory operations: ADD, UPDATE, DELETE, NOOP
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
from openai import OpenAI
import os

from .retrieval_memory import EpisodicMemory, RetrievalAugmentedMemory

from .memory_operations import MemoryOperation

# Import KnowledgeDB
try:
    from ..database.knowledge_db_manager import KnowledgeDatabaseManager
except ImportError:
    KnowledgeDatabaseManager = None


@dataclass
class MemoryDecisionContext:
    """Context dùng để quyết định memory operation"""
    current_info: Dict[str, Any]  # Information từ current dialogue turn
    retrieved_memories: List[EpisodicMemory]  # Memories được retrieve từ RAG
    similarity_scores: List[float]  # Similarity scores với retrieved memories
    dialogue_turn: str  # Current dialogue turn text
    conversation_context: str  # Broader conversation context
    operation: Optional[MemoryOperation] = None
    confidence: float = 0.0
    reasoning: str = ""
    target_memory_id: Optional[str] = None  # ID của memory cần update/delete


class LLMExtractor:
    """Sử dụng LLM để extract key information từ dialogue turns"""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.logger = logging.getLogger("LLMExtractor")
    
    def extract_key_info(self, dialogue_turn: str, context: str = "") -> Dict[str, Any]:
        """
        Extract key information từ dialogue turn sử dụng LLM
        
        Returns:
            Dict chứa extracted information:
            - entities: Named entities
            - intent: User intent
            - key_facts: Important facts
            - topics: Topics discussed
            - sentiment: Sentiment analysis
            - importance: Importance score (0-1)
        """
        
        prompt = f"""
Hãy phân tích dialogue turn sau và trích xuất thông tin quan trọng:

Context: {context}
Dialogue turn: {dialogue_turn}

Trích xuất thông tin theo format JSON:
{{
    "entities": ["entity1", "entity2", ...],
    "intent": "user_intent_description", 
    "key_facts": ["fact1", "fact2", ...],
    "topics": ["topic1", "topic2", ...],
    "sentiment": "positive/negative/neutral",
    "importance": 0.0-1.0,
    "summary": "brief_summary_of_key_points",
    "requires_memory": true/false,
    "memory_type": "factual/personal/preference/procedure/other"
}}

Chỉ trả về JSON, không có text khác.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích dialogue và trích xuất thông tin. Luôn trả về JSON hợp lệ."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                result = json.loads(result_text)
                
                # Validate và fill defaults
                defaults = {
                    "entities": [],
                    "intent": "unknown",
                    "key_facts": [],
                    "topics": [],
                    "sentiment": "neutral",
                    "importance": 0.5,
                    "summary": dialogue_turn[:100] + "...",
                    "requires_memory": True,
                    "memory_type": "other"
                }
                
                for key, default_value in defaults.items():
                    if key not in result:
                        result[key] = default_value
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM JSON response: {e}")
                return self._create_simple_fallback(dialogue_turn)
                
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return self._create_simple_fallback(dialogue_turn)
    
    def _create_simple_fallback(self, dialogue_turn: str) -> Dict[str, Any]:
        """Simple fallback extraction khi LLM fails"""
        
        # Simple rule-based extraction
        words = dialogue_turn.lower().split()
        
        # Simple sentiment
        positive_words = ["tốt", "hay", "thích", "good", "great", "love", "like"]
        negative_words = ["tệ", "xấu", "ghét", "bad", "hate", "terrible", "awful"]
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Simple importance based on length và keywords
        importance = min(len(words) / 50.0, 1.0)  # Longer = more important
        
        important_keywords = ["quan trọng", "cần", "phải", "important", "need", "must"]
        if any(keyword in dialogue_turn.lower() for keyword in important_keywords):
            importance = min(importance + 0.3, 1.0)
        
        return {
            "entities": [],
            "intent": "general_conversation",
            "key_facts": [dialogue_turn[:100] + "..."] if len(dialogue_turn) > 20 else [],
            "topics": ["general"],
            "sentiment": sentiment,
            "importance": importance,
            "summary": dialogue_turn[:100] + "..." if len(dialogue_turn) > 100 else dialogue_turn,
            "requires_memory": len(words) > 5,  # Only remember substantial conversations
            "memory_type": "other"
        }


class IntelligentMemoryManager:
    """
    Intelligent Memory Manager implementing Algorithm 1
    Quyết định thông minh về ADD/UPDATE/DELETE/NOOP operations
    """
    
    def __init__(self,
                 memory_system: RetrievalAugmentedMemory,
                 llm_extractor: LLMExtractor,
                 similarity_threshold_update: float = 0.8,
                 similarity_threshold_delete: float = 0.95,
                 importance_threshold: float = 0.3,
                 max_memory_capacity: int = 5000,
                 knowledge_db_path: str = "data/knowledge_bank.db",
                 enable_knowledge_db: bool = True):
        
        self.memory_system = memory_system
        self.llm_extractor = llm_extractor
        self.similarity_threshold_update = similarity_threshold_update
        self.similarity_threshold_delete = similarity_threshold_delete
        self.importance_threshold = importance_threshold
        self.max_memory_capacity = max_memory_capacity
        
        self.logger = logging.getLogger("MemoryManager")
        
        # Knowledge Database Integration
        self.enable_knowledge_db = enable_knowledge_db and KnowledgeDatabaseManager is not None
        self.knowledge_db = None
        
        if self.enable_knowledge_db:
            try:
                self.knowledge_db = KnowledgeDatabaseManager(knowledge_db_path)
                self.logger.info("Knowledge Database integrated successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Knowledge Database: {e}")
                self.enable_knowledge_db = False
        
        # Statistics
        self.operation_stats = {
            "ADD": 0,
            "UPDATE": 0,
            "DELETE": 0,
            "NOOP": 0
        }
    
    def construct_memory_bank(self, 
                            dialogue_turns: List[str],
                            context: str = "",
                            session_id: str = None,
                            conversation_id: str = None) -> Dict[str, Any]:
        """
        Implementation của Algorithm 1: Memory Bank Construction via Memory Manager
        
        Args:
            dialogue_turns: List các dialogue turns
            context: Conversation context
            
        Returns:
            Dict với results và statistics
        """
        
        results = {
            "operations_performed": [],
            "total_turns_processed": 0,
            "memories_added": 0,
            "memories_updated": 0,
            "memories_deleted": 0,
            "noop_operations": 0,
            "processing_errors": 0
        }
        
        self.logger.info(f"Starting memory bank construction with {len(dialogue_turns)} turns")
        
        for i, dialogue_turn in enumerate(dialogue_turns):
            try:
                # Step 1: Extract key info using LLM
                extracted_info = self.llm_extractor.extract_key_info(dialogue_turn, context)
                
                # Step 1.5: Store extracted knowledge to Knowledge Database (if enabled)
                knowledge_id = None
                if self.enable_knowledge_db and self.knowledge_db:
                    try:
                        knowledge_id = self.knowledge_db.store_extracted_knowledge(
                            extracted_info=extracted_info,
                            dialogue_turn=dialogue_turn,
                            context=context,
                            conversation_id=conversation_id or "unknown",
                            session_id=session_id or "unknown"
                        )
                        self.logger.debug(f"Stored extracted knowledge: {knowledge_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to store extracted knowledge: {e}")
                
                # Skip if not worth remembering
                if not extracted_info.get("requires_memory", True) or extracted_info.get("importance", 0) < self.importance_threshold:
                    operation_result = {
                        "turn_index": i,
                        "operation": MemoryOperation.NOOP,
                        "reasoning": "Below importance threshold or doesn't require memory",
                        "extracted_info": extracted_info,
                        "knowledge_id": knowledge_id
                    }
                    results["operations_performed"].append(operation_result)
                    results["noop_operations"] += 1
                    
                    # Store NOOP operation to Knowledge Database
                    if self.enable_knowledge_db and self.knowledge_db and knowledge_id:
                        try:
                            self.knowledge_db.store_memory_operation(
                                operation_type=MemoryOperation.NOOP,
                                knowledge_id=knowledge_id,
                                confidence=0.9,
                                reasoning="Below importance threshold or doesn't require memory",
                                execution_result={"success": True, "message": "No operation needed"}
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to store NOOP operation: {e}")
                    
                    continue
                
                # Step 2: Retrieve relevant memories using RAG
                query = f"{extracted_info.get('summary', dialogue_turn)} {' '.join(extracted_info.get('key_facts', []))}"
                retrieved_memories = self.memory_system.retrieve_relevant_memories(
                    query, top_k=10  # Increase to get more potential matches
                )
                
                # Step 3: Determine operation using Memory Manager
                decision_context = self._create_decision_context(
                    extracted_info, retrieved_memories, dialogue_turn, context
                )
                
                operation_decision = self.determine_memory_operation(decision_context)
                
                # Step 4: Execute operation
                execution_result = self._execute_operation(
                    operation_decision, extracted_info, dialogue_turn, context
                )
                
                # Store memory operation to Knowledge Database
                if self.enable_knowledge_db and self.knowledge_db and knowledge_id:
                    try:
                        self.knowledge_db.store_memory_operation(
                            operation_type=operation_decision.operation,
                            knowledge_id=knowledge_id,
                            confidence=operation_decision.confidence,
                            reasoning=operation_decision.reasoning,
                            execution_result=execution_result,
                            target_memory_id=operation_decision.target_memory_id
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to store memory operation: {e}")
                
                # Record results
                operation_result = {
                    "turn_index": i,
                    "operation": operation_decision.operation,
                    "confidence": operation_decision.confidence,
                    "reasoning": operation_decision.reasoning,
                    "target_memory_id": operation_decision.target_memory_id,
                    "extracted_info": extracted_info,
                    "execution_result": execution_result,
                    "knowledge_id": knowledge_id
                }
                
                results["operations_performed"].append(operation_result)
                
                # Update counters
                if operation_decision.operation == MemoryOperation.ADD:
                    results["memories_added"] += 1
                elif operation_decision.operation == MemoryOperation.UPDATE:
                    results["memories_updated"] += 1
                elif operation_decision.operation == MemoryOperation.DELETE:
                    results["memories_deleted"] += 1
                else:
                    results["noop_operations"] += 1
                
                self.operation_stats[operation_decision.operation.value] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing turn {i}: {e}")
                results["processing_errors"] += 1
                
                error_result = {
                    "turn_index": i,
                    "operation": MemoryOperation.NOOP,
                    "error": str(e),
                    "dialogue_turn": dialogue_turn
                }
                results["operations_performed"].append(error_result)
            
            results["total_turns_processed"] += 1
        
        self.logger.info(f"Memory bank construction completed: {results}")
        return results
    
    def _create_decision_context(self,
                               extracted_info: Dict[str, Any],
                               retrieved_memories: List[Dict[str, Any]],
                               dialogue_turn: str,
                               context: str) -> MemoryDecisionContext:
        """Tạo context để quyết định memory operation"""
        
        # Convert retrieved memories to EpisodicMemory objects nếu cần
        episodic_memories = []
        similarity_scores = []
        
        for item in retrieved_memories:
            if isinstance(item, dict):
                # Tạo EpisodicMemory từ dict
                memory = EpisodicMemory(
                    id=item.get("id", item.get("memory_id", str(datetime.now().timestamp()))),
                    content=item.get("content", ""),
                    context=item.get("context", ""),
                    importance_score=item.get("importance_score", 1.0),
                    tags=item.get("tags", []),
                    metadata=item.get("metadata", {})
                )
                episodic_memories.append(memory)
                similarity_scores.append(item.get("similarity", item.get("similarity_score", 0.0)))
            elif isinstance(item, EpisodicMemory):
                episodic_memories.append(item)
                similarity_scores.append(0.5)  # Default similarity
        
        return MemoryDecisionContext(
            current_info=extracted_info,
            retrieved_memories=episodic_memories,
            similarity_scores=similarity_scores,
            dialogue_turn=dialogue_turn,
            conversation_context=context
        )
    
    def determine_memory_operation(self, context: MemoryDecisionContext) -> MemoryDecisionContext:
        """
        Core logic để quyết định memory operation dựa trên context
        Implements intelligent decision making cho Algorithm 1
        """
        
        # Nếu không có retrieved memories -> ADD
        if not context.retrieved_memories:
            context.operation = MemoryOperation.ADD
            context.confidence = 0.9
            context.reasoning = "No existing similar memories found - adding new memory"
            return context
        
        # Phân tích similarity với existing memories
        max_similarity = max(context.similarity_scores) if context.similarity_scores else 0.0
        most_similar_memory = None
        most_similar_idx = -1
        
        if context.similarity_scores:
            most_similar_idx = np.argmax(context.similarity_scores)
            most_similar_memory = context.retrieved_memories[most_similar_idx]
        
        # Decision logic dựa trên similarity thresholds
        
        # Case 1: Very high similarity -> potential DELETE or UPDATE
        if max_similarity >= self.similarity_threshold_delete:
            # Check if new info adds value
            new_importance = context.current_info.get("importance", 0.5)
            existing_importance = most_similar_memory.importance_score if most_similar_memory else 0.5
            
            if new_importance <= existing_importance:
                # New info is not more important -> DELETE or NOOP
                if self._should_delete_redundant(context, most_similar_memory):
                    context.operation = MemoryOperation.DELETE
                    context.target_memory_id = most_similar_memory.id
                    context.confidence = 0.8
                    context.reasoning = f"Very similar memory exists with higher importance (sim: {max_similarity:.2f})"
                else:
                    context.operation = MemoryOperation.NOOP
                    context.confidence = 0.9
                    context.reasoning = f"Memory already exists with sufficient quality (sim: {max_similarity:.2f})"
            else:
                # New info is more important -> UPDATE
                context.operation = MemoryOperation.UPDATE
                context.target_memory_id = most_similar_memory.id
                context.confidence = 0.85
                context.reasoning = f"Updating existing memory with more important information (sim: {max_similarity:.2f})"
        
        # Case 2: High similarity -> UPDATE
        elif max_similarity >= self.similarity_threshold_update:
            if self._should_update_memory(context, most_similar_memory):
                context.operation = MemoryOperation.UPDATE
                context.target_memory_id = most_similar_memory.id
                context.confidence = 0.8
                context.reasoning = f"Complementary information to existing memory (sim: {max_similarity:.2f})"
            else:
                context.operation = MemoryOperation.NOOP
                context.confidence = 0.7
                context.reasoning = f"Similar memory exists, no significant new information (sim: {max_similarity:.2f})"
        
        # Case 3: Low similarity -> ADD
        else:
            # Check memory capacity
            if self._is_memory_bank_full():
                # Need to make space - potentially delete least important memory
                if self._should_replace_least_important(context):
                    context.operation = MemoryOperation.ADD  # Will handle cleanup internally
                    context.confidence = 0.7
                    context.reasoning = f"Adding new memory, will cleanup least important (sim: {max_similarity:.2f})"
                else:
                    context.operation = MemoryOperation.NOOP
                    context.confidence = 0.6
                    context.reasoning = "Memory bank full, new information not important enough to replace existing"
            else:
                context.operation = MemoryOperation.ADD
                context.confidence = 0.9
                context.reasoning = f"Sufficiently different from existing memories (sim: {max_similarity:.2f})"
        
        return context
    
    def _should_delete_redundant(self, context: MemoryDecisionContext, existing_memory: EpisodicMemory) -> bool:
        """Quyết định có nên delete redundant memory không"""
        
        # Không delete nếu existing memory có access count cao
        if existing_memory.access_count > 10:
            return False
        
        # Không delete nếu existing memory quá quan trọng
        if existing_memory.importance_score > 0.8:
            return False
        
        # Delete nếu new info có sentiment khác biệt đáng kể
        new_sentiment = context.current_info.get("sentiment", "neutral")
        existing_tags = existing_memory.tags or []
        
        if new_sentiment != "neutral" and f"sentiment_{new_sentiment}" not in existing_tags:
            return False  # Keep both for sentiment diversity
        
        return True
    
    def _should_update_memory(self, context: MemoryDecisionContext, existing_memory: EpisodicMemory) -> bool:
        """Quyết định có nên update existing memory không"""
        
        # Update nếu có new facts
        new_facts = set(context.current_info.get("key_facts", []))
        existing_content = existing_memory.content.lower()
        
        # Check if new facts add information
        new_info_count = 0
        for fact in new_facts:
            if fact.lower() not in existing_content:
                new_info_count += 1
        
        if new_info_count >= 1:  # At least 1 new fact
            return True
        
        # Update nếu sentiment mới và khác
        new_sentiment = context.current_info.get("sentiment", "neutral")
        existing_tags = existing_memory.tags or []
        sentiment_tags = [tag for tag in existing_tags if tag.startswith("sentiment_")]
        
        if new_sentiment != "neutral" and f"sentiment_{new_sentiment}" not in sentiment_tags:
            return True
        
        # Update nếu importance score cao hơn đáng kể
        new_importance = context.current_info.get("importance", 0.5)
        if new_importance > existing_memory.importance_score + 0.2:
            return True
        
        return False
    
    def _is_memory_bank_full(self) -> bool:
        """Check if memory bank is near capacity"""
        try:
            if hasattr(self.memory_system, 'store') and hasattr(self.memory_system.store, 'memories'):
                current_count = len(self.memory_system.store.memories)
                return current_count >= self.max_memory_capacity * 0.9  # 90% capacity
        except:
            pass
        return False
    
    def _should_replace_least_important(self, context: MemoryDecisionContext) -> bool:
        """Quyết định có nên replace least important memory không"""
        new_importance = context.current_info.get("importance", 0.5)
        
        # Chỉ replace nếu new info quan trọng hơn threshold
        return new_importance > 0.6
    
    def _execute_operation(self,
                         decision: MemoryDecisionContext,
                         extracted_info: Dict[str, Any],
                         dialogue_turn: str,
                         context: str) -> Dict[str, Any]:
        """Execute memory operation được quyết định"""
        
        result = {"success": False, "message": "", "memory_id": None}
        
        try:
            if decision.operation == MemoryOperation.ADD:
                # Add new memory với error handling cho dimension mismatch
                try:
                    memory_id = self.memory_system.add_memory(
                        content=dialogue_turn,
                        context=context,
                        tags=extracted_info.get("topics", []) + [f"sentiment_{extracted_info.get('sentiment', 'neutral')}"],
                        importance_score=extracted_info.get("importance", 0.5),
                        metadata={
                            "entities": extracted_info.get("entities", []),
                            "intent": extracted_info.get("intent", ""),
                            "key_facts": extracted_info.get("key_facts", []),
                            "memory_type": extracted_info.get("memory_type", "other"),
                            "operation": "ADD",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                except Exception as e:
                    if "expecting embedding with dimension" in str(e):
                        self.logger.warning(f"Embedding dimension mismatch detected: {e}")
                        # Try again after ChromaDB auto-fixes the collection
                        try:
                            memory_id = self.memory_system.add_memory(
                                content=dialogue_turn,
                                context=context,
                                tags=extracted_info.get("topics", []) + [f"sentiment_{extracted_info.get('sentiment', 'neutral')}"],
                                importance_score=extracted_info.get("importance", 0.5),
                                metadata={
                                    "entities": extracted_info.get("entities", []),
                                    "intent": extracted_info.get("intent", ""),
                                    "key_facts": extracted_info.get("key_facts", []),
                                    "memory_type": extracted_info.get("memory_type", "other"),
                                    "operation": "ADD",
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                        except Exception as retry_error:
                            self.logger.error(f"Failed to add memory after dimension fix: {retry_error}")
                            raise retry_error
                    else:
                        self.logger.error(f"Error adding memory: {e}")
                        raise e
                
                result = {
                    "success": True,
                    "message": "Successfully added new memory",
                    "memory_id": memory_id
                }
                
                # Cleanup if needed
                if self._is_memory_bank_full():
                    self._cleanup_least_important_memories()
            
            elif decision.operation == MemoryOperation.UPDATE:
                # Update existing memory
                success = self._update_memory(
                    decision.target_memory_id,
                    extracted_info,
                    dialogue_turn,
                    context
                )
                
                result = {
                    "success": success,
                    "message": "Successfully updated memory" if success else "Failed to update memory",
                    "memory_id": decision.target_memory_id
                }
            
            elif decision.operation == MemoryOperation.DELETE:
                # Delete memory
                success = self._delete_memory(decision.target_memory_id)
                
                result = {
                    "success": success,
                    "message": "Successfully deleted memory" if success else "Failed to delete memory",
                    "memory_id": decision.target_memory_id
                }
            
            else:  # NOOP
                result = {
                    "success": True,
                    "message": "No operation needed",
                    "memory_id": None
                }
        
        except Exception as e:
            self.logger.error(f"Error executing operation {decision.operation}: {e}")
            result = {
                "success": False,
                "message": f"Operation failed: {str(e)}",
                "memory_id": decision.target_memory_id
            }
        
        return result
    
    def _update_memory(self,
                      memory_id: str,
                      new_info: Dict[str, Any],
                      dialogue_turn: str,
                      context: str) -> bool:
        """Update existing memory với new information"""
        
        try:
            # Get existing memory
            if not hasattr(self.memory_system, 'store') or not hasattr(self.memory_system.store, 'memories'):
                return False
            
            existing_memory = self.memory_system.store.memories.get(memory_id)
            if not existing_memory:
                return False
            
            # Merge information
            merged_content = self._merge_memory_content(
                existing_memory.content,
                dialogue_turn,
                new_info
            )
            
            # Update memory object
            existing_memory.content = merged_content
            existing_memory.context = f"{existing_memory.context}\n{context}".strip()
            existing_memory.importance_score = max(
                existing_memory.importance_score,
                new_info.get("importance", 0.5)
            )
            
            # Merge tags
            new_tags = set(existing_memory.tags or [])
            new_tags.update(new_info.get("topics", []))
            new_tags.add(f"sentiment_{new_info.get('sentiment', 'neutral')}")
            existing_memory.tags = list(new_tags)
            
            # Update metadata
            if not existing_memory.metadata:
                existing_memory.metadata = {}
            
            existing_memory.metadata.update({
                "last_updated": datetime.now().isoformat(),
                "update_count": existing_memory.metadata.get("update_count", 0) + 1,
                "latest_entities": new_info.get("entities", []),
                "latest_key_facts": new_info.get("key_facts", [])
            })
            
            # Update in store (specific implementation depends on store type)
            if hasattr(self.memory_system.store, 'update_memory'):
                return self.memory_system.store.update_memory(existing_memory)
            else:
                # Fallback: re-add to store
                self.memory_system.store.memories[memory_id] = existing_memory
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    def _merge_memory_content(self,
                            existing_content: str,
                            new_content: str,
                            new_info: Dict[str, Any]) -> str:
        """Merge existing và new memory content intelligently"""
        
        # Simple merging strategy
        merged_parts = [existing_content]
        
        # Add new key facts nếu chưa có
        new_facts = new_info.get("key_facts", [])
        existing_lower = existing_content.lower()
        
        for fact in new_facts:
            if fact.lower() not in existing_lower:
                merged_parts.append(f"• {fact}")
        
        # Add new content nếu substantially different
        if len(new_content) > 20 and new_content.lower() not in existing_lower:
            merged_parts.append(f"\nCập nhật: {new_content}")
        
        return "\n".join(merged_parts)
    
    def _delete_memory(self, memory_id: str) -> bool:
        """Delete memory từ store"""
        
        try:
            if hasattr(self.memory_system.store, 'remove_memory'):
                return self.memory_system.store.remove_memory(memory_id)
            elif hasattr(self.memory_system.store, 'memories'):
                if memory_id in self.memory_system.store.memories:
                    del self.memory_system.store.memories[memory_id]
                    return True
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def _cleanup_least_important_memories(self, cleanup_count: int = 10):
        """Cleanup least important memories khi memory bank full"""
        
        try:
            if not hasattr(self.memory_system, 'store') or not hasattr(self.memory_system.store, 'memories'):
                return
            
            memories = list(self.memory_system.store.memories.values())
            if len(memories) <= cleanup_count:
                return
            
            # Sort by importance score và access count
            memories.sort(key=lambda m: (m.importance_score, m.access_count))
            
            # Delete least important memories
            for memory in memories[:cleanup_count]:
                self._delete_memory(memory.id)
                self.logger.info(f"Cleaned up memory {memory.id} (importance: {memory.importance_score})")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup memories: {e}")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Lấy statistics về memory operations"""
        
        total_ops = sum(self.operation_stats.values())
        
        stats = {
            "total_operations": total_ops,
            "operation_counts": self.operation_stats.copy(),
            "operation_percentages": {
                op: (count / total_ops * 100) if total_ops > 0 else 0
                for op, count in self.operation_stats.items()
            },
            "efficiency_metrics": {
                "add_ratio": self.operation_stats["ADD"] / total_ops if total_ops > 0 else 0,
                "update_ratio": self.operation_stats["UPDATE"] / total_ops if total_ops > 0 else 0,
                "noop_ratio": self.operation_stats["NOOP"] / total_ops if total_ops > 0 else 0,
                "memory_bank_size": len(self.memory_system.store.memories) if hasattr(self.memory_system, 'store') and hasattr(self.memory_system.store, 'memories') else 0
            },
            "knowledge_database_enabled": self.enable_knowledge_db
        }
        
        # Add Knowledge Database statistics if available
        if self.enable_knowledge_db and self.knowledge_db:
            try:
                knowledge_stats = self.knowledge_db.get_knowledge_statistics()
                stats["knowledge_database_stats"] = knowledge_stats
            except Exception as e:
                self.logger.warning(f"Failed to get knowledge database stats: {e}")
                stats["knowledge_database_stats"] = {"error": str(e)}
        
        return stats
    
    def get_knowledge_insights(self, session_id: str = None) -> Dict[str, Any]:
        """Lấy insights từ Knowledge Database"""
        
        if not self.enable_knowledge_db or not self.knowledge_db:
            return {"error": "Knowledge Database not available"}
        
        try:
            insights = {
                "database_statistics": self.knowledge_db.get_knowledge_statistics(),
                "recent_knowledge": []
            }
            
            # Get recent knowledge for session
            if session_id:
                recent_knowledge = self.knowledge_db.get_knowledge_by_session(session_id, limit=10)
                insights["recent_knowledge"] = recent_knowledge
                insights["session_summary"] = {
                    "total_entries": len(recent_knowledge),
                    "avg_importance": sum(k.get("importance", 0) for k in recent_knowledge) / max(len(recent_knowledge), 1),
                    "common_topics": self._extract_common_topics(recent_knowledge),
                    "memory_types": self._extract_memory_types(recent_knowledge)
                }
            
            return insights
        
        except Exception as e:
            self.logger.error(f"Failed to get knowledge insights: {e}")
            return {"error": str(e)}
    
    def _extract_common_topics(self, knowledge_entries: List[Dict]) -> List[str]:
        """Extract common topics từ knowledge entries"""
        
        topic_counts = {}
        for entry in knowledge_entries:
            topics = entry.get("topics", [])
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def _extract_memory_types(self, knowledge_entries: List[Dict]) -> Dict[str, int]:
        """Extract memory types distribution"""
        
        type_counts = {}
        for entry in knowledge_entries:
            memory_type = entry.get("memory_type", "other")
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        
        return type_counts
    
    def search_knowledge_bank(self, 
                             query: str, 
                             session_id: str = None,
                             memory_type: str = None,
                             min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """Search trong Knowledge Database"""
        
        if not self.enable_knowledge_db or not self.knowledge_db:
            return []
        
        try:
            return self.knowledge_db.search_knowledge(
                query=query,
                session_id=session_id,
                memory_type=memory_type,
                min_importance=min_importance
            )
        except Exception as e:
            self.logger.error(f"Failed to search knowledge bank: {e}")
            return []
