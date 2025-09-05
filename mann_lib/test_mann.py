#!/usr/bin/env python3
"""
Test script cho MANN system
"""

import asyncio
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from standalone_mann.mann_core import MemoryAugmentedNetwork
from standalone_mann.mann_config import MANNConfig
from standalone_mann.mann_api import MANNClient


async def test_mann_core():
    """Test MANN core functionality"""
    print("🧪 Testing MANN Core...")
    
    # Initialize MANN
    config = MANNConfig()
    mann = MemoryAugmentedNetwork(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        memory_size=config.memory_size,
        memory_dim=config.memory_dim,
        output_size=config.output_size
    )
    
    # Test adding memories
    print("  📝 Adding test memories...")
    memory_ids = []
    
    test_memories = [
        ("Tôi tên là Ngọc", "personal_info", ["name", "personal"]),
        ("Tôi thích lập trình Python", "preference", ["programming", "python"]),
        ("Tôi đang học về AI và Machine Learning", "learning", ["ai", "ml", "learning"]),
        ("Hôm nay trời đẹp", "weather", ["weather", "today"]),
        ("Tôi muốn tạo một chatbot thông minh", "goal", ["chatbot", "ai", "project"])
    ]
    
    for content, context, tags in test_memories:
        memory_id = mann.add_memory(content, context, tags, importance_weight=1.5)
        memory_ids.append(memory_id)
        print(f"    ✅ Added memory: {content[:30]}...")
    
    # Test memory search
    print("  🔍 Testing memory search...")
    search_queries = ["tên", "lập trình", "AI", "chatbot"]
    
    for query in search_queries:
        results = mann.search_memories(query, top_k=3)
        print(f"    Query '{query}': {len(results)} results")
        for i, result in enumerate(results[:2], 1):
            print(f"      {i}. {result['content'][:50]}... (sim: {result.get('similarity', 0):.3f})")
    
    # Test forward pass
    print("  🚀 Testing forward pass...")
    import torch
    input_tensor = torch.randn(1, 1, config.input_size)
    output, memory_info = mann.forward(input_tensor, retrieve_memories=True)
    print(f"    Output shape: {output.shape}")
    print(f"    Retrieved {len(memory_info)} memories")
    
    # Test statistics
    print("  📊 Testing statistics...")
    stats = mann.get_memory_statistics()
    print(f"    Total memories: {stats.get('total_memories', 0)}")
    print(f"    Memory utilization: {stats.get('memory_utilization', 0):.2%}")
    print(f"    Total retrievals: {stats.get('total_retrievals', 0)}")
    
    print("✅ MANN Core tests completed!")


async def test_mann_api():
    """Test MANN API functionality"""
    print("\n🌐 Testing MANN API...")
    
    # Test API client
    async with MANNClient("http://localhost:8000") as client:
        try:
            # Health check
            print("  🏥 Testing health check...")
            health = await client.health_check()
            print(f"    Status: {health.get('status', 'unknown')}")
            print(f"    Memory count: {health.get('memory_count', 0)}")
            
            # Add memory
            print("  📝 Testing add memory...")
            memory_id = await client.add_memory(
                content="Test memory from API",
                context="api_test",
                tags=["test", "api"],
                importance_weight=1.0
            )
            print(f"    Added memory: {memory_id}")
            
            # Search memories
            print("  🔍 Testing search...")
            results = await client.search_memories("test", top_k=3)
            print(f"    Found {len(results)} memories")
            
            # Process query
            print("  🚀 Testing query processing...")
            response = await client.process_query("Hello, how are you?", retrieve_memories=True)
            print(f"    Response: {response.get('output', 'No output')}")
            print(f"    Processing time: {response.get('processing_time', 0):.3f}s")
            print(f"    Retrieved memories: {len(response.get('memory_info', []))}")
            
        except Exception as e:
            print(f"  ❌ API test failed: {e}")
            print("  💡 Make sure API server is running: python run_api.py")
    
    print("✅ MANN API tests completed!")


async def test_chatbot():
    """Test CLI chatbot functionality"""
    print("\n🤖 Testing CLI Chatbot...")
    
    from mann_chatbot import MANNChatbot
    from standalone_mann.mann_config import MANNConfig
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Test conversation
        test_inputs = [
            "Xin chào, tôi tên là Ngọc",
            "Tôi thích lập trình Python",
            "Bạn có nhớ tên tôi không?",
            "Tôi đang học về AI",
            "Bạn biết gì về sở thích của tôi?"
        ]
        
        print("  💬 Testing conversation...")
        for i, user_input in enumerate(test_inputs, 1):
            print(f"    {i}. User: {user_input}")
            response = await chatbot.process_user_input(user_input)
            print(f"       Bot: {response[:100]}...")
        
        # Test memory search
        print("  🔍 Testing memory search...")
        search_results = await chatbot.search_memories("tên", top_k=3)
        print(f"    Found {len(search_results)} memories about 'tên'")
        
        # Test statistics
        print("  📊 Testing statistics...")
        stats = await chatbot.get_memory_statistics()
        print(f"    Total queries: {stats.get('total_queries', 0)}")
        print(f"    Memories created: {stats.get('total_memories_created', 0)}")
        print(f"    Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        
        # Test health check
        print("  🏥 Testing health check...")
        health = await chatbot.health_check()
        print(f"    Health status: {health.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"  ❌ Chatbot test failed: {e}")
    finally:
        await chatbot.shutdown()
    
    print("✅ CLI Chatbot tests completed!")


async def main():
    """Run all tests"""
    print("🧪 MANN System Test Suite")
    print("=" * 50)
    
    try:
        # Test core functionality
        await test_mann_core()
        
        # Test API (only if server is running)
        await test_mann_api()
        
        # Test chatbot
        await test_chatbot()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
