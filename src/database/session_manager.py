"""
Session Manager cho RL Chatbot
Quản lý chat sessions và tích hợp với memory system
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import torch
from dataclasses import dataclass

from .database_manager import DatabaseManager, get_database_manager, ChatMessage
from core.meta_learning import MemoryBankEntry


@dataclass
class SessionContext:
    """Context cho một chat session"""
    session_id: str
    created_at: datetime
    last_updated: datetime
    message_count: int
    memory_bank_size: int
    total_interactions: int
    user_preferences: Dict[str, Any]
    session_metadata: Dict[str, Any]


class SessionManager:
    """
    Quản lý chat sessions và liên kết với memory system
    Đảm bảo memory bank được restore chính xác theo session
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or get_database_manager()
        self.logger = logging.getLogger("SessionManager")
        
        # Cache active sessions để tối ưu performance
        self.active_sessions_cache: Dict[str, SessionContext] = {}
        self.cache_ttl = timedelta(hours=1)  # Cache TTL
        
        # Session limits
        self.max_messages_per_session = 1000
        self.max_sessions_per_user = 10  # Future: user-based limitation
    
    def create_new_session(self, user_id: str = "default", 
                          session_metadata: Dict[str, Any] = None) -> str:
        """
        Tạo session mới cho user
        
        Args:
            user_id: ID của user (future extension)
            session_metadata: Metadata bổ sung cho session
            
        Returns:
            session_id: ID của session mới được tạo
        """
        session_id = str(uuid.uuid4())
        
        # Create session trong database
        self.db_manager.create_session(session_id)
        
        # Initialize session context
        now = datetime.now()
        session_context = SessionContext(
            session_id=session_id,
            created_at=now,
            last_updated=now,
            message_count=0,
            memory_bank_size=0,
            total_interactions=0,
            user_preferences={},
            session_metadata=session_metadata or {}
        )
        
        # Cache session
        self.active_sessions_cache[session_id] = session_context
        
        # Log initial session message
        self.db_manager.add_message(
            session_id=session_id,
            role="system",
            content="Phiên trò chuyện mới đã được bắt đầu",
            metadata={
                "event_type": "session_start",
                "user_id": user_id,
                "session_metadata": session_metadata
            }
        )
        
        self.logger.info(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    def get_or_create_session(self, session_id: str = None, 
                             user_id: str = "default") -> str:
        """
        Lấy session hiện tại hoặc tạo mới nếu không tồn tại
        
        Args:
            session_id: ID session muốn lấy (None để tạo mới)
            user_id: ID của user
            
        Returns:
            session_id: ID của session được sử dụng
        """
        if session_id and self.is_session_valid(session_id):
            self.logger.info(f"Using existing session {session_id}")
            return session_id
        else:
            return self.create_new_session(user_id)
    
    def is_session_valid(self, session_id: str) -> bool:
        """Kiểm tra session có hợp lệ không"""
        try:
            session = self.db_manager.get_session(session_id)
            if not session:
                return False
            
            # Check if session is too old (optional)
            age = datetime.now() - session.last_updated
            if age.days > 30:  # Sessions older than 30 days considered invalid
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking session validity: {e}")
            return False
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """
        Lấy context đầy đủ của session
        
        Args:
            session_id: ID của session
            
        Returns:
            SessionContext hoặc None nếu không tồn tại
        """
        # Check cache first
        if session_id in self.active_sessions_cache:
            cached_context = self.active_sessions_cache[session_id]
            # Check if cache is still valid
            if datetime.now() - cached_context.last_updated < self.cache_ttl:
                return cached_context
        
        # Load from database
        try:
            session = self.db_manager.get_session(session_id)
            if not session:
                return None
            
            # Get additional stats
            messages = self.db_manager.get_chat_history(session_id)
            memory_stats = self.db_manager.get_memory_bank_stats(session_id)
            
            # Extract user preferences từ messages (simple analysis)
            user_preferences = self._extract_user_preferences(messages)
            
            context = SessionContext(
                session_id=session_id,
                created_at=session.created_at,
                last_updated=session.last_updated,
                message_count=session.total_messages,
                memory_bank_size=memory_stats.get('total_entries', 0),
                total_interactions=len(messages),
                user_preferences=user_preferences,
                session_metadata=session.metadata
            )
            
            # Update cache
            self.active_sessions_cache[session_id] = context
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting session context: {e}")
            return None
    
    def add_message_to_session(self, session_id: str, role: str, content: str, 
                              metadata: Dict[str, Any] = None) -> str:
        """
        Thêm message vào session và cập nhật context
        
        Args:
            session_id: ID session
            role: Role của message ('user' hoặc 'assistant')
            content: Nội dung message
            metadata: Metadata bổ sung
            
        Returns:
            message_id: ID của message được tạo
        """
        # Add to database
        message_id = self.db_manager.add_message(session_id, role, content, metadata)
        
        # Update cached context
        if session_id in self.active_sessions_cache:
            context = self.active_sessions_cache[session_id]
            context.message_count += 1
            context.total_interactions += 1
            context.last_updated = datetime.now()
        
        return message_id
    
    def get_conversation_history(self, session_id: str, 
                               limit: int = 50,
                               include_system_messages: bool = False) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử conversation cho session
        
        Args:
            session_id: ID session
            limit: Số lượng messages tối đa
            include_system_messages: Có bao gồm system messages không
            
        Returns:
            List các messages dạng dict
        """
        messages = self.db_manager.get_chat_history(session_id, limit)
        
        conversation = []
        for msg in messages:
            # Skip system messages nếu không yêu cầu
            if not include_system_messages and msg.role == "system":
                continue
            
            conversation.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
                "metadata": msg.metadata
            })
        
        return conversation
    
    def save_memory_bank_for_session(self, session_id: str, 
                                   memory_entries: List[MemoryBankEntry],
                                   timestep: int):
        """
        Lưu memory bank cho session
        
        Args:
            session_id: ID session
            memory_entries: List các MemoryBankEntry
            timestep: Current timestep
        """
        try:
            self.db_manager.save_memory_bank(session_id, memory_entries, timestep)
            
            # Update cached context
            if session_id in self.active_sessions_cache:
                context = self.active_sessions_cache[session_id]
                context.memory_bank_size = len(memory_entries)
                context.last_updated = datetime.now()
            
            self.logger.info(f"Saved memory bank for session {session_id} with {len(memory_entries)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to save memory bank for session {session_id}: {e}")
            raise
    
    def load_memory_bank_for_session(self, session_id: str, 
                                   device: torch.device = None) -> Tuple[List[MemoryBankEntry], int]:
        """
        Load memory bank cho session
        
        Args:
            session_id: ID session
            device: PyTorch device để load tensors
            
        Returns:
            Tuple of (memory_entries, timestep)
        """
        try:
            memory_entries, timestep = self.db_manager.load_memory_bank(session_id, device)
            
            self.logger.info(f"Loaded memory bank for session {session_id} with {len(memory_entries)} entries, timestep {timestep}")
            
            return memory_entries, timestep
            
        except Exception as e:
            self.logger.error(f"Failed to load memory bank for session {session_id}: {e}")
            return [], 0
    
    def get_session_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """Lấy thống kê memory cho session"""
        return self.db_manager.get_memory_bank_stats(session_id)
    
    def _extract_user_preferences(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Extract user preferences từ conversation history
        Simple rule-based approach, có thể mở rộng với ML
        """
        preferences = {
            "preferred_language": "vietnamese",  # Default
            "conversation_style": "casual",      # Default
            "topics_of_interest": [],
            "response_length_preference": "medium"
        }
        
        # Analyze messages để extract preferences
        user_messages = [msg for msg in messages if msg.role == "user"]
        
        if user_messages:
            # Language detection (simple)
            total_content = " ".join([msg.content for msg in user_messages[-10:]])  # Last 10 messages
            if any(word in total_content.lower() for word in ["hello", "thank", "please", "yes", "no"]):
                if any(word in total_content.lower() for word in ["xin", "chào", "cảm ơn", "vâng", "không"]):
                    preferences["preferred_language"] = "mixed"
                else:
                    preferences["preferred_language"] = "english"
            
            # Response length preference
            avg_user_length = sum(len(msg.content) for msg in user_messages) / len(user_messages)
            if avg_user_length < 50:
                preferences["response_length_preference"] = "short"
            elif avg_user_length > 200:
                preferences["response_length_preference"] = "long"
        
        return preferences
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Lấy các sessions gần đây"""
        sessions = self.db_manager.list_sessions(limit)
        
        result = []
        for session in sessions:
            # Get quick stats
            context = self.get_session_context(session.session_id)
            
            result.append({
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_updated": session.last_updated,
                "total_messages": session.total_messages,
                "memory_bank_size": context.memory_bank_size if context else 0,
                "metadata": session.metadata
            })
        
        return result
    
    def cleanup_old_sessions(self, days_threshold: int = 30) -> int:
        """Clean up old sessions"""
        cleaned_count = self.db_manager.cleanup_old_sessions(days_threshold)
        
        # Clear từ cache các sessions đã bị xóa
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=days_threshold)
        
        sessions_to_remove = []
        for session_id, context in self.active_sessions_cache.items():
            if context.last_updated < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions_cache[session_id]
        
        self.logger.info(f"Cleaned up {cleaned_count} sessions and removed {len(sessions_to_remove)} from cache")
        return cleaned_count
    
    def export_session(self, session_id: str, output_path: str):
        """Export session data ra file"""
        self.db_manager.export_session_data(session_id, output_path)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Lấy tóm tắt session cho debugging/monitoring
        
        Returns:
            Dict chứa summary thông tin session
        """
        context = self.get_session_context(session_id)
        if not context:
            return {"error": "Session not found"}
        
        recent_messages = self.db_manager.get_recent_messages(session_id, 5)
        memory_stats = self.get_session_memory_stats(session_id)
        
        return {
            "session_id": session_id,
            "created_at": context.created_at.isoformat(),
            "last_updated": context.last_updated.isoformat(),
            "total_messages": context.message_count,
            "memory_bank_size": context.memory_bank_size,
            "memory_stats": memory_stats,
            "user_preferences": context.user_preferences,
            "recent_messages_count": len(recent_messages),
            "session_age_hours": (datetime.now() - context.created_at).total_seconds() / 3600,
            "last_activity_hours_ago": (datetime.now() - context.last_updated).total_seconds() / 3600
        }


# Singleton pattern
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get singleton session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
