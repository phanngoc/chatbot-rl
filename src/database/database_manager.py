"""
Database Manager cho RL Chatbot
Quản lý session chat history và memory bank persistence
"""

import sqlite3
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid
import os
from pathlib import Path
import logging

# Import memory types
from core.meta_learning import MemoryBankEntry


@dataclass
class ChatSession:
    """Đại diện cho một chat session"""
    session_id: str
    created_at: datetime
    last_updated: datetime
    total_messages: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChatMessage:
    """Đại diện cho một tin nhắn trong chat"""
    message_id: str
    session_id: str
    role: str  # 'user' hoặc 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryBankState:
    """Đại diện cho trạng thái memory bank"""
    session_id: str
    memory_entries: List[Dict[str, Any]]  # Serialized MemoryBankEntry objects
    timestep: int
    created_at: datetime
    last_updated: datetime


class DatabaseManager:
    """
    Quản lý database cho RL Chatbot
    Lưu trữ chat sessions, messages, và memory bank states
    """
    
    def __init__(self, db_path: str = "data/chatbot_database.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("DatabaseManager")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Khởi tạo database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if we need to migrate existing schema
            self._migrate_schema_if_needed(cursor)
            
            # Chat Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    total_messages INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Chat Messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                )
            ''')
            
            # Memory Bank States table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_bank_states (
                    session_id TEXT PRIMARY KEY,
                    memory_entries_blob BLOB NOT NULL,
                    timestep INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                )
            ''')
            
            # Memory Bank Entries table (denormalized để query dễ dàng)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_bank_entries (
                    entry_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    key_vector BLOB NOT NULL,
                    value_vector BLOB NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_accessed INTEGER DEFAULT 0,
                    importance_weight REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                )
            ''')
            
            # Indexes cho performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_session ON memory_bank_entries(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_importance ON memory_bank_entries(importance_weight)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_usage ON memory_bank_entries(usage_count)')
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    def _migrate_schema_if_needed(self, cursor):
        """Migrate schema nếu có constraint cũ"""
        try:
            # Check if table exists và check constraint
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_messages'")
            result = cursor.fetchone()
            
            if result:
                table_sql = result[0]
                # Check if constraint chỉ có user/assistant (missing system)
                if "role IN ('user', 'assistant')" in table_sql and "system" not in table_sql:
                    self.logger.info("Migrating database schema to support system messages...")
                    
                    # Create new table với constraint mới
                    cursor.execute('''
                        CREATE TABLE chat_messages_new (
                            message_id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                            content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            metadata TEXT DEFAULT '{}',
                            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                        )
                    ''')
                    
                    # Copy data từ table cũ
                    cursor.execute('''
                        INSERT INTO chat_messages_new 
                        SELECT * FROM chat_messages
                    ''')
                    
                    # Drop old table và rename new table
                    cursor.execute('DROP TABLE chat_messages')
                    cursor.execute('ALTER TABLE chat_messages_new RENAME TO chat_messages')
                    
                    # Recreate indexes
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp)')
                    
                    self.logger.info("Schema migration completed successfully")
        
        except Exception as e:
            self.logger.warning(f"Schema migration check failed: {e}")
    
    # === Session Management ===
    
    def create_session(self, session_id: str = None) -> str:
        """Tạo chat session mới"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        now = datetime.now()
        session = ChatSession(
            session_id=session_id,
            created_at=now,
            last_updated=now
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_sessions (session_id, created_at, last_updated, total_messages, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                session.created_at.isoformat(),
                session.last_updated.isoformat(),
                session.total_messages,
                json.dumps(session.metadata)
            ))
            conn.commit()
        
        self.logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Lấy thông tin session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, created_at, last_updated, total_messages, metadata
                FROM chat_sessions WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return ChatSession(
                    session_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    last_updated=datetime.fromisoformat(row[2]),
                    total_messages=row[3],
                    metadata=json.loads(row[4])
                )
        return None
    
    def list_sessions(self, limit: int = 100) -> List[ChatSession]:
        """Liệt kê các sessions gần đây"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, created_at, last_updated, total_messages, metadata
                FROM chat_sessions 
                ORDER BY last_updated DESC 
                LIMIT ?
            ''', (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append(ChatSession(
                    session_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    last_updated=datetime.fromisoformat(row[2]),
                    total_messages=row[3],
                    metadata=json.loads(row[4])
                ))
            
            return sessions
    
    def update_session_activity(self, session_id: str):
        """Cập nhật last_updated cho session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE chat_sessions 
                SET last_updated = ?, total_messages = total_messages + 1
                WHERE session_id = ?
            ''', (datetime.now().isoformat(), session_id))
            conn.commit()
    
    # === Message Management ===
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> str:
        """Thêm tin nhắn mới vào session"""
        message_id = str(uuid.uuid4())
        message = ChatMessage(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_messages (message_id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                message.message_id,
                message.session_id,
                message.role,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.metadata)
            ))
            conn.commit()
        
        # Update session activity
        self.update_session_activity(session_id)
        
        return message_id
    
    def get_chat_history(self, session_id: str, limit: int = 100) -> List[ChatMessage]:
        """Lấy lịch sử chat của session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT message_id, session_id, role, content, timestamp, metadata
                FROM chat_messages 
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (session_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                messages.append(ChatMessage(
                    message_id=row[0],
                    session_id=row[1],
                    role=row[2],
                    content=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5])
                ))
            
            return messages
    
    def get_recent_messages(self, session_id: str, count: int = 10) -> List[ChatMessage]:
        """Lấy các tin nhắn gần đây nhất"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT message_id, session_id, role, content, timestamp, metadata
                FROM chat_messages 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, count))
            
            messages = []
            for row in cursor.fetchall():
                messages.append(ChatMessage(
                    message_id=row[0],
                    session_id=row[1],
                    role=row[2],
                    content=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5])
                ))
            
            # Reverse để có thứ tự thời gian tăng dần
            return list(reversed(messages))
    
    # === Memory Bank Management ===
    
    def save_memory_bank(self, session_id: str, memory_entries: List[MemoryBankEntry], 
                        timestep: int):
        """Lưu memory bank state cho session"""
        now = datetime.now()
        
        # Serialize memory entries
        serialized_entries = []
        for entry in memory_entries:
            serialized_entry = {
                'key': entry.key.cpu().numpy().tobytes(),
                'value': entry.value.cpu().numpy().tobytes(),
                'usage_count': entry.usage_count,
                'last_accessed': entry.last_accessed,
                'importance_weight': entry.importance_weight,
                'key_shape': list(entry.key.shape),
                'value_shape': list(entry.value.shape)
            }
            serialized_entries.append(serialized_entry)
        
        # Pickle the entire structure để preserve complex data
        memory_blob = pickle.dumps(serialized_entries)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Upsert memory bank state
            cursor.execute('''
                INSERT OR REPLACE INTO memory_bank_states 
                (session_id, memory_entries_blob, timestep, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                memory_blob,
                timestep,
                now.isoformat(),
                now.isoformat()
            ))
            
            # Also save individual entries cho query purposes
            # Clear existing entries first
            cursor.execute('DELETE FROM memory_bank_entries WHERE session_id = ?', (session_id,))
            
            for i, entry in enumerate(memory_entries):
                entry_id = f"{session_id}_{i}"
                cursor.execute('''
                    INSERT INTO memory_bank_entries 
                    (entry_id, session_id, key_vector, value_vector, usage_count, 
                     last_accessed, importance_weight, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry_id,
                    session_id,
                    entry.key.cpu().numpy().tobytes(),
                    entry.value.cpu().numpy().tobytes(),
                    entry.usage_count,
                    entry.last_accessed,
                    entry.importance_weight,
                    now.isoformat(),
                    now.isoformat()
                ))
            
            conn.commit()
        
        self.logger.info(f"Saved memory bank for session {session_id} with {len(memory_entries)} entries")
    
    def load_memory_bank(self, session_id: str, device: torch.device = None) -> Tuple[List[MemoryBankEntry], int]:
        """Load memory bank state cho session"""
        if device is None:
            device = torch.device('cpu')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT memory_entries_blob, timestep
                FROM memory_bank_states 
                WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return [], 0
            
            memory_blob, timestep = row
            
            try:
                # Deserialize memory entries
                serialized_entries = pickle.loads(memory_blob)
                memory_entries = []
                
                for entry_data in serialized_entries:
                    # Reconstruct tensors
                    key_bytes = entry_data['key']
                    value_bytes = entry_data['value']
                    key_shape = entry_data['key_shape']
                    value_shape = entry_data['value_shape']
                    
                    key_array = np.frombuffer(key_bytes, dtype=np.float32).reshape(key_shape)
                    value_array = np.frombuffer(value_bytes, dtype=np.float32).reshape(value_shape)
                    
                    key_tensor = torch.from_numpy(key_array).to(device)
                    value_tensor = torch.from_numpy(value_array).to(device)
                    
                    entry = MemoryBankEntry(
                        key=key_tensor,
                        value=value_tensor,
                        usage_count=entry_data['usage_count'],
                        last_accessed=entry_data['last_accessed'],
                        importance_weight=entry_data['importance_weight']
                    )
                    memory_entries.append(entry)
                
                self.logger.info(f"Loaded memory bank for session {session_id} with {len(memory_entries)} entries")
                return memory_entries, timestep
                
            except Exception as e:
                self.logger.error(f"Failed to load memory bank for session {session_id}: {e}")
                return [], 0
    
    def get_memory_bank_stats(self, session_id: str) -> Dict[str, Any]:
        """Lấy thống kê memory bank cho session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as total_entries,
                       AVG(usage_count) as avg_usage_count,
                       MAX(usage_count) as max_usage_count,
                       AVG(importance_weight) as avg_importance,
                       MAX(importance_weight) as max_importance,
                       MIN(last_accessed) as oldest_access,
                       MAX(last_accessed) as newest_access
                FROM memory_bank_entries 
                WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    'total_entries': row[0],
                    'avg_usage_count': row[1] or 0,
                    'max_usage_count': row[2] or 0,
                    'avg_importance': row[3] or 0,
                    'max_importance': row[4] or 0,
                    'oldest_access': row[5] or 0,
                    'newest_access': row[6] or 0
                }
            else:
                return {
                    'total_entries': 0,
                    'avg_usage_count': 0,
                    'max_usage_count': 0,
                    'avg_importance': 0,
                    'max_importance': 0,
                    'oldest_access': 0,
                    'newest_access': 0
                }
    
    # === Maintenance ===
    
    def cleanup_old_sessions(self, days_threshold: int = 30) -> int:
        """Xóa các sessions và data cũ"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get sessions to delete
            cursor.execute('''
                SELECT session_id FROM chat_sessions 
                WHERE last_updated < ?
            ''', (cutoff_date.isoformat(),))
            
            old_sessions = [row[0] for row in cursor.fetchall()]
            
            if old_sessions:
                # Delete related data
                for session_id in old_sessions:
                    cursor.execute('DELETE FROM memory_bank_entries WHERE session_id = ?', (session_id,))
                    cursor.execute('DELETE FROM memory_bank_states WHERE session_id = ?', (session_id,))
                    cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
                    cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
                
                conn.commit()
                self.logger.info(f"Cleaned up {len(old_sessions)} old sessions")
            
            return len(old_sessions)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Lấy thống kê tổng quát về database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Session stats
            cursor.execute('SELECT COUNT(*) FROM chat_sessions')
            total_sessions = cursor.fetchone()[0]
            
            # Message stats
            cursor.execute('SELECT COUNT(*) FROM chat_messages')
            total_messages = cursor.fetchone()[0]
            
            # Memory bank stats
            cursor.execute('SELECT COUNT(*) FROM memory_bank_states')
            sessions_with_memory = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM memory_bank_entries')
            total_memory_entries = cursor.fetchone()[0]
            
            # Recent activity
            recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute('SELECT COUNT(*) FROM chat_sessions WHERE last_updated >= ?', (recent_cutoff,))
            recent_active_sessions = cursor.fetchone()[0]
            
            return {
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'sessions_with_memory': sessions_with_memory,
                'total_memory_entries': total_memory_entries,
                'recent_active_sessions': recent_active_sessions,
                'database_file_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            }
    
    def export_session_data(self, session_id: str, output_path: str):
        """Export toàn bộ data của session ra file JSON"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        messages = self.get_chat_history(session_id)
        memory_entries, timestep = self.load_memory_bank(session_id)
        
        export_data = {
            'session': asdict(session),
            'messages': [asdict(msg) for msg in messages],
            'memory_bank': {
                'timestep': timestep,
                'entries_count': len(memory_entries),
                'stats': self.get_memory_bank_stats(session_id)
            },
            'exported_at': datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        for msg in export_data['messages']:
            msg['timestamp'] = msg['timestamp'].isoformat()
        
        export_data['session']['created_at'] = export_data['session']['created_at'].isoformat()
        export_data['session']['last_updated'] = export_data['session']['last_updated'].isoformat()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Exported session {session_id} data to {output_path}")


# Singleton instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get singleton database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
