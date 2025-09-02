#!/usr/bin/env python3
"""
Test script cho episodic experiences UI
Tạo sample data để test giao diện
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.rl_chatbot import RLChatbotAgent
from datetime import datetime

def create_sample_data():
    """Tạo sample episodic experiences để test UI"""
    
    print("🧪 Creating sample episodic experiences...")
    
    # Create agent
    config = {
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    agent = RLChatbotAgent(config=config)
    
    # Start a session
    session_id = agent.start_session(user_id="test_ui")
    print(f"✅ Started session: {session_id}")
    
    # Sample conversations với different rewards
    sample_conversations = [
        {
            "user": "Hello, how are you?",
            "bot": "I'm doing great! How can I help you today?",
            "reward": 0.8,
            "feedback": "Friendly response"
        },
        {
            "user": "What's the weather like?",
            "bot": "I don't have access to current weather data, but I can help you find weather information.",
            "reward": 0.3,
            "feedback": "Honest but not very helpful"
        },
        {
            "user": "Tell me about AI",
            "bot": "Artificial Intelligence is a fascinating field that involves creating systems that can perform tasks requiring human intelligence.",
            "reward": 0.9,
            "feedback": "Great informative response"
        },
        {
            "user": "What's 2+2?",
            "bot": "2+2 equals 4. This is basic arithmetic.",
            "reward": 1.0,
            "feedback": "Perfect answer"
        },
        {
            "user": "Can you help me with homework?",
            "bot": "I'd be happy to help! What subject is your homework about?",
            "reward": 0.7,
            "feedback": "Helpful and engaging"
        },
        {
            "user": "I'm feeling sad",
            "bot": "I'm sorry to hear you're feeling sad. Would you like to talk about what's bothering you?",
            "reward": 0.6,
            "feedback": "Empathetic response"
        },
        {
            "user": "What's your favorite color?",
            "bot": "As an AI, I don't have personal preferences, but I can discuss colors and their meanings!",
            "reward": 0.4,
            "feedback": "Accurate but could be more engaging"
        },
        {
            "user": "Explain quantum physics",
            "bot": "Quantum physics is the study of matter and energy at the smallest scales, where particles behave very differently than in our everyday world.",
            "reward": 0.85,
            "feedback": "Good simplified explanation"
        }
    ]
    
    # Process conversations và tạo episodic experiences
    for i, conv in enumerate(sample_conversations):
        try:
            # Simulate conversation
            result = agent.process_message(conv["user"])
            print(f"✅ Processed conversation {i+1}: '{conv['user'][:30]}...'")
            
            # Add experience to meta-learning system manually với custom data
            agent.meta_learning_system.store_episodic_experience_with_autosave(
                context=conv["user"],
                response=conv["bot"],
                reward=conv["reward"],
                user_feedback=conv["feedback"]
            )
            
            # Also add to experience buffer
            from core.experience_replay import Experience
            experience = Experience(
                state=conv["user"],
                action=conv["bot"],
                reward=conv["reward"],
                next_state="",
                timestamp=datetime.now(),
                conversation_id=session_id,
                user_feedback=conv["feedback"]
            )
            agent.experience_buffer.add_experience(experience)
            
        except Exception as e:
            print(f"⚠️  Error processing conversation {i+1}: {e}")
    
    # Force save memory bank
    success = agent.force_save_memory()
    print(f"💾 Force save memory: {'Success' if success else 'Failed'}")
    
    # Print statistics
    meta_stats = agent.meta_learning_system.get_system_statistics()
    buffer_stats = agent.experience_buffer.get_statistics()
    
    print("\n📊 Sample Data Created:")
    print(f"   • Meta-learning experiences: {len(agent.meta_learning_system.experience_buffer)}")
    print(f"   • Memory bank entries: {meta_stats['memory_bank']['total_memories']}")
    print(f"   • Experience buffer: {buffer_stats['total_experiences']}")
    print(f"   • Session ID: {session_id}")
    
    return session_id

def test_ui_data_access():
    """Test việc truy cập data cho UI"""
    
    print("\n🔍 Testing UI data access...")
    
    try:
        # Create agent
        agent = RLChatbotAgent()
        
        # Test accessing episodic experiences
        meta_stats = agent.meta_learning_system.get_system_statistics()
        experience_buffer = agent.meta_learning_system.experience_buffer
        
        print(f"✅ Meta-learning stats: {len(experience_buffer)} experiences")
        print(f"✅ Memory bank: {meta_stats['memory_bank']['total_memories']} entries")
        
        # Test memory search
        if len(experience_buffer) > 0:
            memories = agent.meta_learning_system.select_relevant_memories(
                "hello", top_k=3
            )
            print(f"✅ Memory search: Found {len(memories)} relevant memories")
        
        # Test experience buffer
        buffer_stats = agent.experience_buffer.get_statistics()
        print(f"✅ Experience buffer: {buffer_stats['total_experiences']} experiences")
        
        print("🎉 UI data access test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ UI data access test failed: {e}")
        return False

def main():
    """Main function"""
    
    print("Episodic Experiences UI Test")
    print("=" * 40)
    
    # Create sample data
    session_id = create_sample_data()
    
    # Test UI data access
    success = test_ui_data_access()
    
    print("\n" + "=" * 40)
    print("🎯 TEST RESULTS:")
    print(f"   Sample data created: ✅")
    print(f"   UI data access: {'✅' if success else '❌'}")
    print(f"   Session for testing: {session_id}")
    
    print("\n🚀 Ready to test Streamlit UI!")
    print("   Run: streamlit run src/app.py")
    print("   Navigate to: 📚 Quản lý Session and 🔍 Khám phá bộ nhớ")

if __name__ == "__main__":
    main()
