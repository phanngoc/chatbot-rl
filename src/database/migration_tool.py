"""
Migration Tool để chuyển đổi dữ liệu hiện tại sang database mới
Giúp preserve existing data và migrate sang session-based structure
"""

import os
import json
import pickle
import torch
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .database_manager import get_database_manager
from .session_manager import get_session_manager
from core.meta_learning import MemoryBankEntry


class DataMigrationTool:
    """Tool để migrate existing data sang database mới"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataMigrationTool")
        self.db_manager = get_database_manager()
        self.session_manager = get_session_manager()
        
        # Paths to existing data files
        self.data_dir = Path("data")
        self.existing_files = {
            "agent_state": self.data_dir / "agent_state.json",
            "meta_learning": self.data_dir / "agent_state_meta_learning.pt",
            "model": self.data_dir / "agent_state_model.pt",
            "retrieval_memories": self.data_dir / "agent_state_retrieval_memories.json",
            "temporal_weights": self.data_dir / "agent_state_temporal_weights.json",
            "consolidated_knowledge": self.data_dir / "agent_state_consolidated_knowledge.json",
            "experience_buffer": self.data_dir / "experience_buffer.pkl"
        }
    
    def run_full_migration(self, create_default_session: bool = True) -> Dict[str, Any]:
        """
        Chạy full migration cho tất cả existing data
        
        Args:
            create_default_session: Có tạo default session cho data không có session
            
        Returns:
            Dict với results của migration
        """
        migration_results = {
            "sessions_created": 0,
            "messages_migrated": 0,
            "memory_entries_migrated": 0,
            "experiences_migrated": 0,
            "errors": [],
            "warnings": [],
            "migration_started": datetime.now().isoformat()
        }
        
        self.logger.info("Starting full data migration...")
        
        try:
            # 1. Migrate existing agent state và conversation history
            if self.existing_files["agent_state"].exists():
                result = self._migrate_agent_state()
                migration_results.update(result)
            
            # 2. Migrate meta-learning memory bank
            if create_default_session:
                default_session = self._create_default_session()
                migration_results["sessions_created"] += 1
                
                # Migrate memory bank to default session
                memory_result = self._migrate_memory_bank(default_session)
                migration_results.update(memory_result)
            
            # 3. Migrate experience buffer
            if self.existing_files["experience_buffer"].exists():
                exp_result = self._migrate_experience_buffer()
                migration_results.update(exp_result)
            
            # 4. Backup original files
            self._backup_original_files()
            
            migration_results["migration_completed"] = datetime.now().isoformat()
            migration_results["success"] = True
            
            self.logger.info(f"Migration completed successfully: {migration_results}")
            
        except Exception as e:
            migration_results["errors"].append(f"Migration failed: {str(e)}")
            migration_results["success"] = False
            self.logger.error(f"Migration failed: {e}")
        
        return migration_results
    
    def _create_default_session(self) -> str:
        """Tạo default session cho existing data"""
        session_id = self.session_manager.create_new_session(
            user_id="legacy_user",
            session_metadata={
                "migration_source": "legacy_data",
                "migration_timestamp": datetime.now().isoformat(),
                "description": "Migrated from existing chatbot data"
            }
        )
        
        self.logger.info(f"Created default session for migration: {session_id}")
        return session_id
    
    def _migrate_agent_state(self) -> Dict[str, Any]:
        """Migrate agent state và conversation history"""
        results = {
            "conversations_migrated": 0,
            "messages_migrated": 0,
            "agent_state_errors": []
        }
        
        try:
            with open(self.existing_files["agent_state"], 'r', encoding='utf-8') as f:
                agent_state = json.load(f)
            
            # Extract conversation history
            conversation_history = agent_state.get("conversation_history", [])
            
            if conversation_history:
                # Create session cho conversation history
                session_id = self.session_manager.create_new_session(
                    user_id="legacy_conversation",
                    session_metadata={
                        "migration_source": "agent_state_conversation_history",
                        "original_conversation_id": agent_state.get("current_conversation_id"),
                        "migration_timestamp": datetime.now().isoformat()
                    }
                )
                
                # Migrate messages
                for i, msg in enumerate(conversation_history):
                    try:
                        # Parse message format (có thể khác nhau)
                        if isinstance(msg, dict):
                            user_msg = msg.get("user_message", "")
                            bot_msg = msg.get("bot_response", "")
                            timestamp = msg.get("timestamp")
                            
                            if user_msg:
                                self.session_manager.add_message_to_session(
                                    session_id, "user", user_msg,
                                    metadata={
                                        "migration_index": i,
                                        "original_timestamp": timestamp
                                    }
                                )
                                results["messages_migrated"] += 1
                            
                            if bot_msg:
                                self.session_manager.add_message_to_session(
                                    session_id, "assistant", bot_msg,
                                    metadata={
                                        "migration_index": i,
                                        "original_timestamp": timestamp,
                                        "experience_id": msg.get("experience_id")
                                    }
                                )
                                results["messages_migrated"] += 1
                    
                    except Exception as e:
                        results["agent_state_errors"].append(f"Message {i}: {str(e)}")
                
                results["conversations_migrated"] = 1
                self.logger.info(f"Migrated conversation history with {len(conversation_history)} entries to session {session_id}")
        
        except Exception as e:
            results["agent_state_errors"].append(f"Failed to load agent state: {str(e)}")
            self.logger.error(f"Failed to migrate agent state: {e}")
        
        return results
    
    def _migrate_memory_bank(self, session_id: str) -> Dict[str, Any]:
        """Migrate meta-learning memory bank"""
        results = {
            "memory_entries_migrated": 0,
            "memory_migration_errors": []
        }
        
        try:
            # Try to load existing memory bank data
            meta_learning_file = self.existing_files["meta_learning"]
            
            if meta_learning_file.exists():
                # Load PyTorch model state
                checkpoint = torch.load(meta_learning_file, map_location='cpu')
                
                if "memory_bank" in checkpoint:
                    memory_bank_data = checkpoint["memory_bank"]
                    timestep = checkpoint.get("timestep", 0)
                    
                    # Convert to MemoryBankEntry objects
                    memory_entries = []
                    for i, entry_data in enumerate(memory_bank_data):
                        try:
                            if isinstance(entry_data, dict):
                                # Reconstruct MemoryBankEntry
                                key_tensor = entry_data.get("key", torch.randn(128))
                                value_tensor = entry_data.get("value", torch.randn(128))
                                
                                # Ensure tensors are proper format
                                if not isinstance(key_tensor, torch.Tensor):
                                    key_tensor = torch.tensor(key_tensor)
                                if not isinstance(value_tensor, torch.Tensor):
                                    value_tensor = torch.tensor(value_tensor)
                                
                                entry = MemoryBankEntry(
                                    key=key_tensor,
                                    value=value_tensor,
                                    usage_count=entry_data.get("usage_count", 0),
                                    last_accessed=entry_data.get("last_accessed", 0),
                                    importance_weight=entry_data.get("importance_weight", 1.0)
                                )
                                memory_entries.append(entry)
                                results["memory_entries_migrated"] += 1
                        
                        except Exception as e:
                            results["memory_migration_errors"].append(f"Entry {i}: {str(e)}")
                    
                    # Save to database
                    if memory_entries:
                        self.session_manager.save_memory_bank_for_session(
                            session_id, memory_entries, timestep
                        )
                        self.logger.info(f"Migrated {len(memory_entries)} memory entries to session {session_id}")
                
        except Exception as e:
            results["memory_migration_errors"].append(f"Failed to load meta-learning data: {str(e)}")
            self.logger.error(f"Failed to migrate memory bank: {e}")
        
        return results
    
    def _migrate_experience_buffer(self) -> Dict[str, Any]:
        """Migrate experience buffer data"""
        results = {
            "experiences_migrated": 0,
            "experience_migration_errors": []
        }
        
        try:
            with open(self.existing_files["experience_buffer"], 'rb') as f:
                buffer_data = pickle.load(f)
            
            # Extract experiences
            experiences = []
            if isinstance(buffer_data, dict):
                experiences = buffer_data.get("buffer", [])
            elif isinstance(buffer_data, list):
                experiences = buffer_data
            
            # Create session cho experiences
            if experiences:
                session_id = self.session_manager.create_new_session(
                    user_id="legacy_experiences",
                    session_metadata={
                        "migration_source": "experience_buffer",
                        "migration_timestamp": datetime.now().isoformat(),
                        "total_experiences": len(experiences)
                    }
                )
                
                # Migrate experiences as conversation pairs
                for i, exp in enumerate(experiences):
                    try:
                        if hasattr(exp, 'state') and hasattr(exp, 'action'):
                            # Add as conversation pair
                            self.session_manager.add_message_to_session(
                                session_id, "user", exp.state,
                                metadata={
                                    "migration_source": "experience_buffer",
                                    "experience_index": i,
                                    "reward": getattr(exp, 'reward', 0.0),
                                    "original_timestamp": getattr(exp, 'timestamp', None)
                                }
                            )
                            
                            self.session_manager.add_message_to_session(
                                session_id, "assistant", exp.action,
                                metadata={
                                    "migration_source": "experience_buffer", 
                                    "experience_index": i,
                                    "reward": getattr(exp, 'reward', 0.0),
                                    "conversation_id": getattr(exp, 'conversation_id', None)
                                }
                            )
                            
                            results["experiences_migrated"] += 1
                    
                    except Exception as e:
                        results["experience_migration_errors"].append(f"Experience {i}: {str(e)}")
                
                self.logger.info(f"Migrated {results['experiences_migrated']} experiences to session {session_id}")
        
        except Exception as e:
            results["experience_migration_errors"].append(f"Failed to load experience buffer: {str(e)}")
            self.logger.error(f"Failed to migrate experience buffer: {e}")
        
        return results
    
    def _backup_original_files(self):
        """Backup original files trước khi migration"""
        backup_dir = self.data_dir / "backup_pre_migration"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, file_path in self.existing_files.items():
            if file_path.exists():
                try:
                    backup_path = backup_dir / f"{name}_{timestamp}{file_path.suffix}"
                    # Copy file
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    self.logger.info(f"Backed up {file_path} to {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to backup {file_path}: {e}")
    
    def verify_migration(self, session_id: str = None) -> Dict[str, Any]:
        """Verify migration results"""
        verification = {
            "database_stats": self.db_manager.get_database_stats(),
            "session_summaries": [],
            "memory_bank_stats": [],
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Get all sessions if none specified
        if session_id:
            sessions = [session_id]
        else:
            recent_sessions = self.session_manager.get_recent_sessions(20)
            sessions = [s["session_id"] for s in recent_sessions]
        
        for sid in sessions:
            try:
                summary = self.session_manager.get_session_summary(sid)
                verification["session_summaries"].append(summary)
                
                memory_stats = self.session_manager.get_session_memory_stats(sid)
                verification["memory_bank_stats"].append({
                    "session_id": sid,
                    "stats": memory_stats
                })
            except Exception as e:
                self.logger.warning(f"Failed to verify session {sid}: {e}")
        
        return verification
    
    def rollback_migration(self) -> bool:
        """Rollback migration (for emergency)"""
        try:
            # This is a destructive operation - clear new database
            # Restore from backup files
            
            self.logger.warning("Rollback not fully implemented - manual restoration required")
            return False
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False


def run_migration_cli():
    """Command line interface cho migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate RL Chatbot data to new database structure")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing migration")
    parser.add_argument("--create-session", action="store_true", default=True, help="Create default session")
    parser.add_argument("--backup", action="store_true", default=True, help="Backup original files")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    migration_tool = DataMigrationTool()
    
    if args.verify_only:
        print("Verifying migration...")
        verification = migration_tool.verify_migration()
        print(json.dumps(verification, indent=2, ensure_ascii=False))
    else:
        print("Starting migration...")
        results = migration_tool.run_full_migration(
            create_default_session=args.create_session
        )
        print("Migration Results:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Run verification
        print("\nVerifying migration...")
        verification = migration_tool.verify_migration()
        print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_migration_cli()
