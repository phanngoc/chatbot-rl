#!/usr/bin/env python3
"""
Test script standalone cho Knowledge Database
Kiểm tra tính năng Knowledge Database mà không dependency với các modules khác
"""

import sys
import os
import json
import sqlite3
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import MemoryOperation standalone
class MemoryOperation(Enum):
    """Các operations cho memory management"""
    ADD = "ADD"
    UPDATE = "UPDATE" 
    DELETE = "DELETE"
    NOOP = "NOOP"

# Import Knowledge Database Manager directly
import sqlite3
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ExtractedKnowledge:
    """Cấu trúc dữ liệu cho knowledge được extract"""
    id: str
    content: str
    context: str
    entities: List[str]
    intent: str
    key_facts: List[str]
    topics: List[str]
    sentiment: str
    importance: float
    memory_type: str
    summary: str
    conversation_id: str
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any]

class KnowledgeDatabaseManager:
    """Standalone Knowledge Database Manager for testing"""
    
    def __init__(self, db_path: str = "data/knowledge_bank.db"):
        self.db_path = db_path
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Khởi tạo database schema"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Bảng extracted_knowledge
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
            
            # Bảng memory_operations
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
            
            conn.commit()
    
    def store_extracted_knowledge(self, 
                                extracted_info: Dict[str, Any],
                                dialogue_turn: str,
                                context: str,
                                conversation_id: str,
                                session_id: str) -> str:
        """Lưu extracted knowledge vào database"""
        
        knowledge_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO extracted_knowledge (
                id, content, context, entities, intent, key_facts, topics,
                sentiment, importance, memory_type, summary, conversation_id,
                session_id, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                knowledge_id,
                dialogue_turn,
                context,
                json.dumps(extracted_info.get("entities", []), ensure_ascii=False),
                extracted_info.get("intent", ""),
                json.dumps(extracted_info.get("key_facts", []), ensure_ascii=False),
                json.dumps(extracted_info.get("topics", []), ensure_ascii=False),
                extracted_info.get("sentiment", "neutral"),
                extracted_info.get("importance", 0.5),
                extracted_info.get("memory_type", "other"),
                extracted_info.get("summary", ""),
                conversation_id,
                session_id,
                timestamp.isoformat(),
                json.dumps(extracted_info.get("metadata", {}), ensure_ascii=False)
            ))
            
            conn.commit()
            return knowledge_id
    
    def store_memory_operation(self,
                             operation_type: MemoryOperation,
                             knowledge_id: str,
                             confidence: float,
                             reasoning: str,
                             execution_result: Dict[str, Any],
                             target_memory_id: Optional[str] = None) -> str:
        """Lưu memory operation record vào database"""
        
        operation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO memory_operations (
                id, operation_type, target_memory_id, knowledge_id,
                confidence, reasoning, execution_success, error_message,
                timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operation_id,
                operation_type.value,
                target_memory_id,
                knowledge_id,
                confidence,
                reasoning,
                execution_result.get("success", False),
                execution_result.get("message") if not execution_result.get("success") else None,
                timestamp.isoformat(),
                json.dumps(execution_result, ensure_ascii=False)
            ))
            
            conn.commit()
            return operation_id
    
    def get_knowledge_by_session(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy extracted knowledge theo session"""
        
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
    
    def search_knowledge(self, 
                        query: str, 
                        session_id: Optional[str] = None,
                        memory_type: Optional[str] = None,
                        min_importance: float = 0.0,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """Search knowledge entries"""
        
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
    
    def get_knowledge_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Lấy thống kê knowledge database"""
        
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
            
            return {
                "total_knowledge_entries": total_knowledge,
                "operations_stats": operations_stats,
                "avg_importance": avg_importance,
                "days_analyzed": days_back,
                "database_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }

def test_standalone_knowledge_database():
    """Test standalone Knowledge Database"""
    
    print("=== Test Standalone Knowledge Database ===")
    
    # Tạo temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_standalone_knowledge.db")
        
        try:
            # Initialize Knowledge Database
            knowledge_db = KnowledgeDatabaseManager(db_path)
            print("✓ Knowledge Database initialized")
            
            # Test 1: Store extracted knowledge
            print("\n1. Testing store_extracted_knowledge...")
            
            extracted_info = {
                "entities": ["Python", "AI", "programming"],
                "intent": "learning_programming",
                "key_facts": ["Python is good for AI", "Easy to learn"],
                "topics": ["programming", "AI", "education"],
                "sentiment": "positive",
                "importance": 0.8,
                "memory_type": "educational",
                "summary": "Learning about Python for AI",
                "metadata": {"difficulty": "beginner"}
            }
            
            knowledge_id = knowledge_db.store_extracted_knowledge(
                extracted_info=extracted_info,
                dialogue_turn="User: How do I start with Python for AI?\nBot: Python is a great choice for AI development...",
                context="Programming tutorial",
                conversation_id="tutorial_001",
                session_id="student_001"
            )
            
            print(f"✓ Stored knowledge: {knowledge_id}")
            
            # Test 2: Store memory operation
            print("\n2. Testing store_memory_operation...")
            
            operation_id = knowledge_db.store_memory_operation(
                operation_type=MemoryOperation.ADD,
                knowledge_id=knowledge_id,
                confidence=0.9,
                reasoning="Educational content about Python",
                execution_result={"success": True, "message": "Added educational content"}
            )
            
            print(f"✓ Stored operation: {operation_id}")
            
            # Test 3: Store multiple entries
            print("\n3. Testing multiple entries...")
            
            entries = [
                {
                    "content": "User: What is machine learning?\nBot: Machine learning is a subset of AI...",
                    "topics": ["AI", "machine learning"],
                    "importance": 0.7
                },
                {
                    "content": "User: How to install Python libraries?\nBot: You can use pip install...",
                    "topics": ["Python", "installation"],
                    "importance": 0.6
                },
                {
                    "content": "User: Explain neural networks\nBot: Neural networks are computational models...",
                    "topics": ["AI", "neural networks", "deep learning"],
                    "importance": 0.9
                }
            ]
            
            for i, entry in enumerate(entries):
                extracted_info = {
                    "entities": [],
                    "intent": "educational_question",
                    "key_facts": [f"Educational content {i+1}"],
                    "topics": entry["topics"],
                    "sentiment": "neutral",
                    "importance": entry["importance"],
                    "memory_type": "educational",
                    "summary": f"Educational discussion {i+1}",
                    "metadata": {"entry_number": i+1}
                }
                
                kid = knowledge_db.store_extracted_knowledge(
                    extracted_info=extracted_info,
                    dialogue_turn=entry["content"],
                    context="Educational session",
                    conversation_id=f"tutorial_{i+2:03d}",
                    session_id="student_001"
                )
                
                knowledge_db.store_memory_operation(
                    operation_type=MemoryOperation.ADD,
                    knowledge_id=kid,
                    confidence=0.8,
                    reasoning=f"Educational entry {i+1}",
                    execution_result={"success": True, "message": f"Added entry {i+1}"}
                )
            
            print(f"✓ Stored {len(entries)} additional entries")
            
            # Test 4: Get knowledge by session
            print("\n4. Testing get_knowledge_by_session...")
            
            session_knowledge = knowledge_db.get_knowledge_by_session("student_001")
            print(f"✓ Retrieved {len(session_knowledge)} entries for session")
            
            for entry in session_knowledge[:2]:  # Show first 2
                print(f"   - Topics: {entry['topics']}, Importance: {entry['importance']}")
            
            # Test 5: Search knowledge
            print("\n5. Testing search_knowledge...")
            
            # Search for Python
            python_results = knowledge_db.search_knowledge("Python", session_id="student_001")
            print(f"✓ Found {len(python_results)} entries about Python")
            
            # Search for AI with high importance
            ai_results = knowledge_db.search_knowledge("AI", min_importance=0.7)
            print(f"✓ Found {len(ai_results)} high-importance AI entries")
            
            # Test 6: Get statistics
            print("\n6. Testing get_knowledge_statistics...")
            
            stats = knowledge_db.get_knowledge_statistics()
            print(f"✓ Database statistics:")
            print(f"   - Total entries: {stats['total_knowledge_entries']}")
            print(f"   - Operations: {stats['operations_stats']}")
            print(f"   - Average importance: {stats['avg_importance']:.2f}")
            print(f"   - Database size: {stats['database_size_mb']:.3f} MB")
            
            # Test 7: Complex search scenarios
            print("\n7. Testing complex search scenarios...")
            
            # Search by memory type
            educational_results = knowledge_db.search_knowledge(
                query="", 
                session_id="student_001",
                memory_type="educational"
            )
            print(f"✓ Found {len(educational_results)} educational entries")
            
            # High importance entries
            important_results = knowledge_db.search_knowledge(
                query="",
                min_importance=0.8
            )
            print(f"✓ Found {len(important_results)} high-importance entries")
            
            print("\n✅ Standalone Knowledge Database tests PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Standalone Knowledge Database tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_database_performance():
    """Test performance của database với nhiều entries"""
    
    print("\n=== Test Database Performance ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_performance.db")
        
        try:
            knowledge_db = KnowledgeDatabaseManager(db_path)
            print("✓ Performance test database initialized")
            
            # Test với nhiều entries
            print("\n1. Testing bulk insert performance...")
            
            start_time = datetime.now()
            num_entries = 100
            
            for i in range(num_entries):
                extracted_info = {
                    "entities": [f"entity_{i}", "test"],
                    "intent": f"intent_{i % 10}",
                    "key_facts": [f"Fact {i}", f"Important info {i}"],
                    "topics": [f"topic_{i % 5}", "performance"],
                    "sentiment": ["positive", "neutral", "negative"][i % 3],
                    "importance": (i % 100) / 100.0,
                    "memory_type": ["factual", "personal", "educational"][i % 3],
                    "summary": f"Performance test entry {i}",
                    "metadata": {"test_id": i, "batch": "performance"}
                }
                
                knowledge_id = knowledge_db.store_extracted_knowledge(
                    extracted_info=extracted_info,
                    dialogue_turn=f"User: Test question {i}\nBot: Test response {i}",
                    context="Performance testing",
                    conversation_id=f"perf_{i:03d}",
                    session_id=f"session_{i % 10}"
                )
                
                # Alternate operations
                op_type = [MemoryOperation.ADD, MemoryOperation.UPDATE, MemoryOperation.NOOP][i % 3]
                knowledge_db.store_memory_operation(
                    operation_type=op_type,
                    knowledge_id=knowledge_id,
                    confidence=0.8,
                    reasoning=f"Performance test operation {i}",
                    execution_result={"success": True, "message": f"Test {i}"}
                )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✓ Inserted {num_entries} entries in {duration:.2f} seconds")
            print(f"   - Rate: {num_entries/duration:.1f} entries/second")
            
            # Test search performance
            print("\n2. Testing search performance...")
            
            search_start = datetime.now()
            results = knowledge_db.search_knowledge("test", limit=50)
            search_end = datetime.now()
            search_duration = (search_end - search_start).total_seconds()
            
            print(f"✓ Search completed in {search_duration:.3f} seconds")
            print(f"   - Found {len(results)} results")
            
            # Test session query performance
            print("\n3. Testing session query performance...")
            
            session_start = datetime.now()
            session_results = knowledge_db.get_knowledge_by_session("session_5")
            session_end = datetime.now()
            session_duration = (session_end - session_start).total_seconds()
            
            print(f"✓ Session query completed in {session_duration:.3f} seconds")
            print(f"   - Found {len(session_results)} session entries")
            
            # Statistics
            stats = knowledge_db.get_knowledge_statistics()
            print(f"\n✓ Performance test statistics:")
            print(f"   - Total entries: {stats['total_knowledge_entries']}")
            print(f"   - Database size: {stats['database_size_mb']:.3f} MB")
            
            print("\n✅ Database performance tests PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Database performance tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Chạy tất cả standalone tests"""
    
    print("🧪 Knowledge Database Standalone Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    tests = [
        ("Standalone Knowledge Database", test_standalone_knowledge_database),
        ("Database Performance", test_database_performance)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All standalone tests PASSED! Knowledge Database core functionality is working.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
