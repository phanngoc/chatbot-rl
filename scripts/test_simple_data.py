#!/usr/bin/env python3
"""
Simple test script tạo sample data mà không cần OpenAI API
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def create_simple_sample_data():
    """Tạo sample data đơn giản"""
    
    print("🧪 Creating simple sample data...")
    
    try:
        from database.database_manager import get_database_manager
        from database.session_manager import get_session_manager
        from core.meta_learning import MetaLearningEpisodicSystem, MemoryBankEntry
        import torch
        
        # Create database components
        db_manager = get_database_manager()
        session_manager = get_session_manager()
        
        # Create test session
        session_id = session_manager.create_new_session(
            user_id="test_ui_user",
            session_metadata={
                "purpose": "UI testing",
                "created_by": "test_script"
            }
        )
        print(f"✅ Created session: {session_id}")
        
        # Add sample messages
        conversations = [
            ("user", "Hello, how are you?"),
            ("assistant", "I'm doing great! How can I help you today?"),
            ("user", "What's the weather like?"),
            ("assistant", "I don't have access to current weather data."),
            ("user", "Tell me about AI"),
            ("assistant", "AI is a fascinating field involving machine learning."),
            ("user", "What's 2+2?"),
            ("assistant", "2+2 equals 4."),
            ("user", "Can you help me with homework?"),
            ("assistant", "I'd be happy to help! What subject?"),
        ]
        
        for role, content in conversations:
            msg_id = session_manager.add_message_to_session(
                session_id, role, content,
                metadata={"test_data": True}
            )
            print(f"✅ Added {role} message: {content[:30]}...")
        
        # Create meta-learning system với sample data
        meta_system = MetaLearningEpisodicSystem(
            input_size=768,
            memory_size=10,
            memory_dim=128,
            session_id=session_id
        )
        
        # Add sample episodic experiences
        sample_experiences = [
            {
                "context": "Hello, how are you?",
                "response": "I'm doing great! How can I help you today?",
                "reward": 0.8,
                "user_feedback": "Friendly response"
            },
            {
                "context": "What's the weather like?", 
                "response": "I don't have access to current weather data.",
                "reward": 0.3,
                "user_feedback": "Honest but not helpful"
            },
            {
                "context": "Tell me about AI",
                "response": "AI is a fascinating field involving machine learning.",
                "reward": 0.9,
                "user_feedback": "Great explanation"
            },
            {
                "context": "What's 2+2?",
                "response": "2+2 equals 4.",
                "reward": 1.0,
                "user_feedback": "Perfect answer"
            },
            {
                "context": "Can you help me with homework?",
                "response": "I'd be happy to help! What subject?",
                "reward": 0.7,
                "user_feedback": "Helpful"
            }
        ]
        
        for exp in sample_experiences:
            meta_system.store_episodic_experience_with_autosave(
                context=exp["context"],
                response=exp["response"],
                reward=exp["reward"],
                user_feedback=exp["user_feedback"]
            )
            print(f"✅ Added episodic experience: {exp['context'][:30]}...")
        
        # Create sample memory bank entries
        sample_memories = [
            MemoryBankEntry(
                key=torch.randn(128),
                value=torch.randn(128),
                usage_count=i+1,
                last_accessed=i*10,
                importance_weight=1.0 + i*0.2
            ) for i in range(5)
        ]
        
        session_manager.save_memory_bank_for_session(
            session_id, sample_memories, timestep=100
        )
        print(f"✅ Saved {len(sample_memories)} memory bank entries")
        
        # Verify data
        stats = db_manager.get_database_stats()
        print(f"\n📊 Created sample data:")
        print(f"   • Sessions: {stats['total_sessions']}")
        print(f"   • Messages: {stats['total_messages']}")
        print(f"   • Memory entries: {stats['total_memory_entries']}")
        print(f"   • Session ID: {session_id}")
        
        return session_id
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    
    print("Simple Sample Data Creator")
    print("=" * 40)
    
    session_id = create_simple_sample_data()
    
    if session_id:
        print("\n🎉 Sample data created successfully!")
        print(f"   Session ID: {session_id}")
        print("\n🚀 Ready to test Streamlit UI!")
        print("   Run: streamlit run src/app.py")
        print("   Navigate to: 📚 Quản lý Session")
        print("   Note: Some features may not work without OpenAI API key")
    else:
        print("\n❌ Failed to create sample data")

if __name__ == "__main__":
    main()
