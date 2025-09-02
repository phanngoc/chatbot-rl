#!/usr/bin/env python3
"""
Test script cho database integration
Kiểm tra tất cả functionality của session-based memory system
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database.database_manager import get_database_manager
from database.session_manager import get_session_manager
from core.meta_learning import MetaLearningEpisodicSystem, MemoryBankEntry
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_database_creation():
    """Test database creation và basic operations"""
    print("🔍 Testing database creation...")
    
    db_manager = get_database_manager()
    
    # Test create session
    session_id = db_manager.create_session()
    print(f"✅ Created session: {session_id}")
    
    # Test add messages
    user_msg_id = db_manager.add_message(session_id, "user", "Hello, how are you?")
    bot_msg_id = db_manager.add_message(session_id, "assistant", "I'm doing well, thank you!")
    print(f"✅ Added messages: {user_msg_id}, {bot_msg_id}")
    
    # Test get messages
    messages = db_manager.get_chat_history(session_id)
    print(f"✅ Retrieved {len(messages)} messages")
    
    # Test memory bank storage
    memory_entries = [
        MemoryBankEntry(
            key=torch.randn(128),
            value=torch.randn(128),
            usage_count=1,
            last_accessed=100,
            importance_weight=1.5
        ) for _ in range(5)
    ]
    
    db_manager.save_memory_bank(session_id, memory_entries, timestep=100)
    print(f"✅ Saved {len(memory_entries)} memory entries")
    
    # Test load memory bank
    loaded_entries, timestep = db_manager.load_memory_bank(session_id)
    print(f"✅ Loaded {len(loaded_entries)} memory entries, timestep: {timestep}")
    
    # Test stats
    stats = db_manager.get_database_stats()
    print(f"✅ Database stats: {stats}")
    
    return session_id

def test_session_manager():
    """Test session manager functionality"""
    print("\n🔍 Testing session manager...")
    
    session_manager = get_session_manager()
    
    # Create new session
    session_id = session_manager.create_new_session(
        user_id="test_user",
        session_metadata={"test": "data"}
    )
    print(f"✅ Created session: {session_id}")
    
    # Add some messages
    session_manager.add_message_to_session(session_id, "user", "Test message 1")
    session_manager.add_message_to_session(session_id, "assistant", "Test response 1")
    print("✅ Added test messages")
    
    # Get session context
    context = session_manager.get_session_context(session_id)
    print(f"✅ Session context: {context.message_count if context else 'None'} messages")
    
    # Get conversation history
    history = session_manager.get_conversation_history(session_id)
    print(f"✅ Conversation history: {len(history)} messages")
    
    # Test memory operations
    memory_entries = [
        MemoryBankEntry(
            key=torch.randn(128),
            value=torch.randn(128),
            usage_count=i,
            last_accessed=i * 10,
            importance_weight=1.0 + i * 0.1
        ) for i in range(3)
    ]
    
    session_manager.save_memory_bank_for_session(session_id, memory_entries, 50)
    print(f"✅ Saved memory bank for session")
    
    loaded_entries, timestep = session_manager.load_memory_bank_for_session(session_id)
    print(f"✅ Loaded memory bank: {len(loaded_entries)} entries, timestep: {timestep}")
    
    # Get session summary
    summary = session_manager.get_session_summary(session_id)
    print(f"✅ Session summary: {summary}")
    
    return session_id

def test_meta_learning_integration():
    """Test meta-learning với database integration"""
    print("\n🔍 Testing meta-learning integration...")
    
    # Create session
    session_manager = get_session_manager()
    session_id = session_manager.create_new_session(user_id="meta_test")
    
    # Create meta-learning system với session
    meta_system = MetaLearningEpisodicSystem(
        input_size=768,
        memory_size=100,
        memory_dim=128,
        session_id=session_id
    )
    
    print(f"✅ Created meta-learning system for session: {session_id}")
    
    # Store some experiences
    for i in range(5):
        meta_system.store_episodic_experience_with_autosave(
            context=f"Test context {i}",
            response=f"Test response {i}",
            reward=0.5 + i * 0.1
        )
    
    print("✅ Stored episodic experiences with auto-save")
    
    # Test memory retrieval
    memories = meta_system.select_relevant_memories("test query", top_k=3)
    print(f"✅ Retrieved {len(memories)} relevant memories")
    
    # Force save memory
    success = meta_system.save_memory_to_database(force_save=True)
    print(f"✅ Force save memory: {'Success' if success else 'Failed'}")
    
    # Get system statistics
    stats = meta_system.get_system_statistics()
    print(f"✅ System statistics: {stats}")
    
    return session_id

def test_agent_integration():
    """Test full agent integration"""
    print("\n🔍 Testing agent integration...")
    
    try:
        from agents.rl_chatbot import RLChatbotAgent
        
        # Create agent
        config = {
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        agent = RLChatbotAgent(config=config)
        print("✅ Created RL Chatbot Agent")
        
        # Start session
        session_id = agent.start_session(user_id="integration_test")
        print(f"✅ Started session: {session_id}")
        
        # Process some messages
        response1 = agent.process_message("Hello, how are you?")
        print(f"✅ Processed message 1: {response1['response'][:50]}...")
        
        response2 = agent.process_message("Tell me about AI")
        print(f"✅ Processed message 2: {response2['response'][:50]}...")
        
        # Test session management methods
        summary = agent.get_session_summary()
        print(f"✅ Session summary: {summary['total_messages']} messages")
        
        # Test database stats
        db_stats = agent.get_database_stats()
        print(f"✅ Database stats: {db_stats.get('total_sessions', 0)} sessions")
        
        # Test force save
        save_success = agent.force_save_memory()
        print(f"✅ Force save memory: {'Success' if save_success else 'Failed'}")
        
        return session_id
        
    except ImportError as e:
        print(f"⚠️  Agent integration test skipped: {e}")
        return None

def test_migration_tool():
    """Test migration tool"""
    print("\n🔍 Testing migration tool...")
    
    try:
        from database.migration_tool import DataMigrationTool
        
        migration_tool = DataMigrationTool()
        print("✅ Created migration tool")
        
        # Run verification (safe)
        verification = migration_tool.verify_migration()
        print(f"✅ Migration verification: {verification['database_stats']}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Migration tool test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        db_manager = get_database_manager()
        
        # Cleanup old sessions
        cleaned = db_manager.cleanup_old_sessions(days_threshold=0)  # Remove all
        print(f"✅ Cleaned up {cleaned} test sessions")
        
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting database integration tests...\n")
    
    test_results = {}
    
    try:
        # Test basic database operations
        session_id_1 = test_database_creation()
        test_results["database_creation"] = "✅ PASSED"
        
        # Test session manager
        session_id_2 = test_session_manager()
        test_results["session_manager"] = "✅ PASSED"
        
        # Test meta-learning integration
        session_id_3 = test_meta_learning_integration()
        test_results["meta_learning"] = "✅ PASSED"
        
        # Test agent integration
        session_id_4 = test_agent_integration()
        test_results["agent_integration"] = "✅ PASSED" if session_id_4 else "⚠️  SKIPPED"
        
        # Test migration tool
        migration_success = test_migration_tool()
        test_results["migration_tool"] = "✅ PASSED" if migration_success else "⚠️  FAILED"
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        test_results["overall"] = "❌ FAILED"
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    for test_name, result in test_results.items():
        print(f"{test_name:.<40} {result}")
    
    # Final database stats
    try:
        db_manager = get_database_manager()
        final_stats = db_manager.get_database_stats()
        print(f"\n📈 Final database stats:")
        print(f"   Sessions: {final_stats.get('total_sessions', 0)}")
        print(f"   Messages: {final_stats.get('total_messages', 0)}")
        print(f"   Memory entries: {final_stats.get('total_memory_entries', 0)}")
        print(f"   Database size: {final_stats.get('database_file_size', 0) / 1024:.1f} KB")
    except Exception as e:
        print(f"⚠️  Could not get final stats: {e}")
    
    print("\n🎉 All tests completed!")
    
    # Ask if user wants to cleanup
    cleanup_response = input("\n🗑️  Do you want to cleanup test data? (y/N): ")
    if cleanup_response.lower().startswith('y'):
        cleanup_test_data()

if __name__ == "__main__":
    main()
