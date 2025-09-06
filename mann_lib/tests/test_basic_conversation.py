#!/usr/bin/env python3
"""
Test: Basic Conversation
Tests basic chatbot conversation functionality
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def test_basic_conversation():
    """Test basic conversation flow"""
    print("🎭 Test: Basic Conversation")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Test conversation sequences
        conversations = [
            "Xin chào, tôi tên là Ngọc",
            "Tôi 25 tuổi và đang làm việc tại Hà Nội",
            "Tôi thích lập trình Python và học về AI",
            "Bạn có nhớ tên tôi không?",
            "Tôi đang học về Machine Learning",
            "Bạn biết gì về sở thích của tôi?",
            "Tôi muốn tạo một chatbot thông minh",
            "Hôm nay trời đẹp, tôi đi dạo công viên"
        ]
        
        print(f"📝 Testing {len(conversations)} conversation turns...")
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 User {i}: {user_input}")
            
            start_time = time.time()
            response = await chatbot.process_user_input(user_input)
            processing_time = time.time() - start_time
            
            print(f"🤖 Bot: {response}")
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            
            # Small delay for demo effect
            await asyncio.sleep(0.5)
        
        # Show final statistics
        stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  Total queries: {stats.get('total_queries', 0)}")
        print(f"  Memories created: {stats.get('total_memories_created', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        print(f"  Total retrievals: {stats.get('total_retrievals', 0)}")
        print(f"  Total writes: {stats.get('total_writes', 0)}")
        print(f"  Memory matrix norm: {stats.get('memory_matrix_norm', 0):.4f}")
        
        print("\n✅ Basic Conversation Test Complete!")
        
    finally:
        await chatbot.shutdown()


if __name__ == "__main__":
    asyncio.run(test_basic_conversation())