"""
Knowledge Database Manager
Quản lý việc lưu trữ extracted knowledge từ Memory Bank vào SQLite database
"""

import sqlite3
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os

from ..memory.memory_operations import MemoryOperation


@dataclass
class ExtractedKnowledge:
    """Cấu trúc dữ liệu cho knowledge được extract"""
    id: str
    content: str  # Original dialogue turn content
    context: str  # Conversation context
    entities: List[str]  # Named entities
    intent: str  # User intent
    key_facts: List[str]  # Important facts
    topics: List[str]  # Discussion topics
    sentiment: str  # Sentiment analysis result
    importance: float  # Importance score (0-1)
    memory_type: str  # Type of memory (factual/personal/preference/procedure/other)
    summary: str  # Brief summary
    conversation_id: str  # Reference to conversation
    session_id: str  # Reference to session
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class MemoryOperationRecord:
    """Record của memory operation được thực hiện"""
    id: str
    operation_type: str  # ADD/UPDATE/DELETE/NOOP
    target_memory_id: Optional[str]  # ID của memory bị ảnh hưởng
    knowledge_id: str  # Reference to ExtractedKnowledge
    confidence: float  # Confidence score của operation
    reasoning: str  # Lý do thực hiện operation
    execution_success: bool  # Kết quả thực hiện
    error_message: Optional[str]  # Error message nếu có
    timestamp: datetime
    metadata: Dict[str, Any]


class KnowledgeDatabaseManager:
    """
    Manager để lưu trữ và quản lý extracted knowledge trong SQLite database
    """
    
    def __init__(self, db_path: str = "data/knowledge_bank.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("KnowledgeDB")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Khởi tạo database schema"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Bảng extracted_knowledge - lưu kiến thức đã extract
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_knowledge (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    context TEXT,
                    entities TEXT,  -- JSON array
                    intent TEXT,
                    key_facts TEXT,  -- JSON array
                    topics TEXT,  -- JSON array
                    sentiment TEXT,
                    importance REAL,
                    memory_type TEXT,
                    summary TEXT,
                    conversation_id TEXT,
                    session_id TEXT,
                    timestamp TEXT,
                    metadata TEXT,  -- JSON object
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Bảng memory_operations - lưu lịch sử operations
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_operations (
                    id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    target_memory_id TEXT,
                    knowledge_id TEXT NOT NULL,
                    confidence REAL,
                    reasoning TEXT,
                    execution_success BOOLEAN,
                    error_message TEXT,
                    timestamp TEXT,
                    metadata TEXT,  -- JSON object
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (knowledge_id) REFERENCES extracted_knowledge (id)
                )
                """)
                
                # Bảng conversation_turns - lưu dialogue turns với extracted info
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    conversation_id TEXT,
                    turn_index INTEGER,
                    user_message TEXT,
                    bot_response TEXT,
                    turn_context TEXT,
                    knowledge_id TEXT,  -- Reference to extracted knowledge
                    timestamp TEXT,
                    metadata TEXT,  -- JSON object
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (knowledge_id) REFERENCES extracted_knowledge (id)
                )
                """)
                
                # Bảng knowledge_statistics - thống kê
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_date DATE,
                    total_knowledge_entries INTEGER,
                    add_operations INTEGER,
                    update_operations INTEGER,
                    delete_operations INTEGER,
                    noop_operations INTEGER,
                    avg_importance REAL,
                    common_topics TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Indexes cho performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_session ON extracted_knowledge (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_conversation ON extracted_knowledge (conversation_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_timestamp ON extracted_knowledge (timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_importance ON extracted_knowledge (importance)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_memory_type ON extracted_knowledge (memory_type)")
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_type ON memory_operations (operation_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_knowledge ON memory_operations (knowledge_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON memory_operations (timestamp)")
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_knowledge ON conversation_turns (knowledge_id)")
                
                conn.commit()
                self.logger.info(f"Knowledge database initialized at: {self.db_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge database: {e}")
            raise
    
    def store_extracted_knowledge(self, 
                                extracted_info: Dict[str, Any],
                                dialogue_turn: str,
                                context: str,
                                conversation_id: str,
                                session_id: str) -> str:
        """
        Lưu extracted knowledge vào database
        
        Returns:
            knowledge_id: ID của knowledge entry đã được tạo
        """
        
        knowledge_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        knowledge = ExtractedKnowledge(
            id=knowledge_id,
            content=dialogue_turn,
            context=context,
            entities=extracted_info.get("entities", []),
            intent=extracted_info.get("intent", ""),
            key_facts=extracted_info.get("key_facts", []),
            topics=extracted_info.get("topics", []),
            sentiment=extracted_info.get("sentiment", "neutral"),
            importance=extracted_info.get("importance", 0.5),
            memory_type=extracted_info.get("memory_type", "other"),
            summary=extracted_info.get("summary", ""),
            conversation_id=conversation_id,
            session_id=session_id,
            timestamp=timestamp,
            metadata=extracted_info.get("metadata", {})
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO extracted_knowledge (
                    id, content, context, entities, intent, key_facts, topics,
                    sentiment, importance, memory_type, summary, conversation_id,
                    session_id, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge.id,
                    knowledge.content,
                    knowledge.context,
                    json.dumps(knowledge.entities, ensure_ascii=False),
                    knowledge.intent,
                    json.dumps(knowledge.key_facts, ensure_ascii=False),
                    json.dumps(knowledge.topics, ensure_ascii=False),
                    knowledge.sentiment,
                    knowledge.importance,
                    knowledge.memory_type,
                    knowledge.summary,
                    knowledge.conversation_id,
                    knowledge.session_id,
                    knowledge.timestamp.isoformat(),
                    json.dumps(knowledge.metadata, ensure_ascii=False)
                ))
                
                conn.commit()
                self.logger.info(f"Stored extracted knowledge: {knowledge_id}")
                return knowledge_id
        
        except Exception as e:
            self.logger.error(f"Failed to store extracted knowledge: {e}")
            raise
    
    def store_memory_operation(self,
                             operation_type: MemoryOperation,
                             knowledge_id: str,
                             confidence: float,
                             reasoning: str,
                             execution_result: Dict[str, Any],
                             target_memory_id: Optional[str] = None) -> str:
        """
        Lưu memory operation record vào database
        
        Returns:
            operation_id: ID của operation record
        """
        
        operation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        operation_record = MemoryOperationRecord(
            id=operation_id,
            operation_type=operation_type.value,
            target_memory_id=target_memory_id,
            knowledge_id=knowledge_id,
            confidence=confidence,
            reasoning=reasoning,
            execution_success=execution_result.get("success", False),
            error_message=execution_result.get("message") if not execution_result.get("success") else None,
            timestamp=timestamp,
            metadata=execution_result
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO memory_operations (
                    id, operation_type, target_memory_id, knowledge_id,
                    confidence, reasoning, execution_success, error_message,
                    timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    operation_record.id,
                    operation_record.operation_type,
                    operation_record.target_memory_id,
                    operation_record.knowledge_id,
                    operation_record.confidence,
                    operation_record.reasoning,
                    operation_record.execution_success,
                    operation_record.error_message,
                    operation_record.timestamp.isoformat(),
                    json.dumps(operation_record.metadata, ensure_ascii=False)
                ))
                
                conn.commit()
                self.logger.info(f"Stored memory operation: {operation_id} ({operation_type.value})")
                return operation_id
        
        except Exception as e:
            self.logger.error(f"Failed to store memory operation: {e}")
            raise
    
    def store_conversation_turn(self,
                              session_id: str,
                              conversation_id: str,
                              turn_index: int,
                              user_message: str,
                              bot_response: str,
                              turn_context: str,
                              knowledge_id: str) -> str:
        """Lưu conversation turn với reference đến extracted knowledge"""
        
        turn_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO conversation_turns (
                    id, session_id, conversation_id, turn_index,
                    user_message, bot_response, turn_context,
                    knowledge_id, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    turn_id,
                    session_id,
                    conversation_id,
                    turn_index,
                    user_message,
                    bot_response,
                    turn_context,
                    knowledge_id,
                    timestamp.isoformat(),
                    json.dumps({"turn_length": len(user_message) + len(bot_response)})
                ))
                
                conn.commit()
                self.logger.info(f"Stored conversation turn: {turn_id}")
                return turn_id
        
        except Exception as e:
            self.logger.error(f"Failed to store conversation turn: {e}")
            raise
    
    def get_knowledge_by_session(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy extracted knowledge theo session"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM extracted_knowledge 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """, (session_id, limit))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                results = []
                for row in rows:
                    knowledge = dict(zip(columns, row))
                    # Parse JSON fields
                    knowledge["entities"] = json.loads(knowledge["entities"] or "[]")
                    knowledge["key_facts"] = json.loads(knowledge["key_facts"] or "[]")
                    knowledge["topics"] = json.loads(knowledge["topics"] or "[]")
                    knowledge["metadata"] = json.loads(knowledge["metadata"] or "{}")
                    results.append(knowledge)
                
                return results
        
        except Exception as e:
            self.logger.error(f"Failed to get knowledge by session: {e}")
            return []
    
    def get_memory_operations_by_knowledge(self, knowledge_id: str) -> List[Dict[str, Any]]:
        """Lấy tất cả memory operations cho một knowledge entry"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM memory_operations 
                WHERE knowledge_id = ? 
                ORDER BY timestamp ASC
                """, (knowledge_id,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                results = []
                for row in rows:
                    operation = dict(zip(columns, row))
                    operation["metadata"] = json.loads(operation["metadata"] or "{}")
                    results.append(operation)
                
                return results
        
        except Exception as e:
            self.logger.error(f"Failed to get memory operations: {e}")
            return []
    
    def get_knowledge_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Lấy thống kê knowledge database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total knowledge entries
                cursor.execute("SELECT COUNT(*) FROM extracted_knowledge")
                total_knowledge = cursor.fetchone()[0]
                
                # Operations statistics
                cursor.execute("""
                SELECT operation_type, COUNT(*) 
                FROM memory_operations 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY operation_type
                """.format(days_back))
                
                operations_stats = dict(cursor.fetchall())
                
                # Average importance
                cursor.execute("""
                SELECT AVG(importance) 
                FROM extracted_knowledge 
                WHERE timestamp >= datetime('now', '-{} days')
                """.format(days_back))
                
                avg_importance = cursor.fetchone()[0] or 0.0
                
                # Common topics
                cursor.execute("""
                SELECT topics 
                FROM extracted_knowledge 
                WHERE timestamp >= datetime('now', '-{} days')
                AND topics != '[]'
                """.format(days_back))
                
                all_topics = []
                for row in cursor.fetchall():
                    topics = json.loads(row[0] or "[]")
                    all_topics.extend(topics)
                
                # Count topic frequencies
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                return {
                    "total_knowledge_entries": total_knowledge,
                    "operations_stats": operations_stats,
                    "avg_importance": avg_importance,
                    "common_topics": common_topics,
                    "days_analyzed": days_back,
                    "database_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
        
        except Exception as e:
            self.logger.error(f"Failed to get knowledge statistics: {e}")
            return {}
    
    def search_knowledge(self, 
                        query: str, 
                        session_id: Optional[str] = None,
                        memory_type: Optional[str] = None,
                        min_importance: float = 0.0,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """Search knowledge entries"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                where_clauses = ["(content LIKE ? OR summary LIKE ? OR key_facts LIKE ?)"]
                params = [f"%{query}%", f"%{query}%", f"%{query}%"]
                
                if session_id:
                    where_clauses.append("session_id = ?")
                    params.append(session_id)
                
                if memory_type:
                    where_clauses.append("memory_type = ?")
                    params.append(memory_type)
                
                if min_importance > 0:
                    where_clauses.append("importance >= ?")
                    params.append(min_importance)
                
                where_clause = " AND ".join(where_clauses)
                params.append(limit)
                
                sql = f"""
                SELECT * FROM extracted_knowledge 
                WHERE {where_clause}
                ORDER BY importance DESC, timestamp DESC 
                LIMIT ?
                """
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                results = []
                for row in rows:
                    knowledge = dict(zip(columns, row))
                    # Parse JSON fields
                    knowledge["entities"] = json.loads(knowledge["entities"] or "[]")
                    knowledge["key_facts"] = json.loads(knowledge["key_facts"] or "[]")
                    knowledge["topics"] = json.loads(knowledge["topics"] or "[]")
                    knowledge["metadata"] = json.loads(knowledge["metadata"] or "{}")
                    results.append(knowledge)
                
                return results
        
        except Exception as e:
            self.logger.error(f"Failed to search knowledge: {e}")
            return []
    
    def update_daily_statistics(self):
        """Cập nhật thống kê hàng ngày"""
        
        try:
            today = datetime.now().date()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if today's stats already exist
                cursor.execute("""
                SELECT id FROM knowledge_statistics 
                WHERE stat_date = ?
                """, (today.isoformat(),))
                
                if cursor.fetchone():
                    return  # Already updated today
                
                # Calculate today's statistics
                stats = self.get_knowledge_statistics(days_back=1)
                
                cursor.execute("""
                INSERT INTO knowledge_statistics (
                    stat_date, total_knowledge_entries, add_operations,
                    update_operations, delete_operations, noop_operations,
                    avg_importance, common_topics, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    today.isoformat(),
                    stats.get("total_knowledge_entries", 0),
                    stats.get("operations_stats", {}).get("ADD", 0),
                    stats.get("operations_stats", {}).get("UPDATE", 0),
                    stats.get("operations_stats", {}).get("DELETE", 0),
                    stats.get("operations_stats", {}).get("NOOP", 0),
                    stats.get("avg_importance", 0.0),
                    json.dumps([topic[0] for topic in stats.get("common_topics", [])[:5]]),
                    json.dumps({"database_size_mb": stats.get("database_size_mb", 0)})
                ))
                
                conn.commit()
                self.logger.info(f"Updated daily statistics for {today}")
        
        except Exception as e:
            self.logger.error(f"Failed to update daily statistics: {e}")
    
    def cleanup_old_data(self, days_threshold: int = 90) -> Dict[str, int]:
        """Cleanup dữ liệu cũ"""
        
        results = {"knowledge_cleaned": 0, "operations_cleaned": 0, "turns_cleaned": 0}
        
        try:
            cutoff_date = datetime.now() - datetime.timedelta(days=days_threshold)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean old knowledge with low importance
                cursor.execute("""
                DELETE FROM extracted_knowledge 
                WHERE timestamp < ? AND importance < 0.3
                """, (cutoff_date.isoformat(),))
                results["knowledge_cleaned"] = cursor.rowcount
                
                # Clean old operations
                cursor.execute("""
                DELETE FROM memory_operations 
                WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                results["operations_cleaned"] = cursor.rowcount
                
                # Clean old conversation turns (keep references intact)
                cursor.execute("""
                DELETE FROM conversation_turns 
                WHERE timestamp < ? AND knowledge_id NOT IN (
                    SELECT id FROM extracted_knowledge
                )
                """, (cutoff_date.isoformat(),))
                results["turns_cleaned"] = cursor.rowcount
                
                conn.commit()
                
                # Vacuum database
                cursor.execute("VACUUM")
                
                self.logger.info(f"Cleaned up old data: {results}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
        
        return results
    
    def export_knowledge_data(self, output_path: str, session_id: Optional[str] = None):
        """Export knowledge data to JSON file"""
        
        try:
            if session_id:
                knowledge_data = self.get_knowledge_by_session(session_id)
            else:
                knowledge_data = self.get_knowledge_by_session("", limit=1000)  # Get all
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "total_entries": len(knowledge_data),
                "knowledge_entries": knowledge_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Exported {len(knowledge_data)} knowledge entries to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to export knowledge data: {e}")
            raise
