#!/usr/bin/env python3
"""
Demo script cho MANN CLI Chatbot
Thể hiện các tính năng chính của hệ thống
"""

import asyncio
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def demo_basic_conversation():
    """Demo cuộc trò chuyện cơ bản"""
    print("🎭 Demo: Basic Conversation")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Simulate conversation
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
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 User {i}: {user_input}")
            
            start_time = time.time()
            response = await chatbot.process_user_input(user_input)
            processing_time = time.time() - start_time
            
            print(f"🤖 Bot: {response}")
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            
            # Small delay for demo effect
            await asyncio.sleep(1)
        
        # Show statistics
        stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Session Statistics:")
        print(f"  Total queries: {stats.get('total_queries', 0)}")
        print(f"  Memories created: {stats.get('total_memories_created', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        print(f"  Total retrievals: {stats.get('total_retrievals', 0)}")
        print(f"  Total writes: {stats.get('total_writes', 0)}")
        print(f"  Memory matrix norm: {stats.get('memory_matrix_norm', 0):.4f}")
        print(f"  Ŵ norm: {stats.get('W_hat_norm', 0):.4f}")
        print(f"  V̂ norm: {stats.get('V_hat_norm', 0):.4f}")
        
    finally:
        await chatbot.shutdown()


async def demo_memory_search():
    """Demo tìm kiếm memory"""
    print("\n🔍 Demo: Memory Search")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Add some test memories first
        test_memories = [
            "Tôi tên là Ngọc, 25 tuổi",
            "Tôi thích lập trình Python và JavaScript",
            "Tôi đang học về AI và Machine Learning",
            "Tôi sống ở Hà Nội và làm việc tại công ty ABC",
            "Tôi thích đọc sách và chơi game",
            "Tôi muốn trở thành một AI Engineer",
            "Hôm nay tôi đi mua sắm ở trung tâm thương mại",
            "Tôi có một con mèo tên là Mimi"
        ]
        
        print("📝 Adding test memories...")
        for memory in test_memories:
            await chatbot.process_user_input(memory)
            await asyncio.sleep(0.5)
        
        # Search for different topics
        search_queries = [
            "tên",
            "lập trình",
            "AI",
            "Hà Nội",
            "mèo",
            "sở thích"
        ]
        
        print("\n🔍 Searching memories...")
        for query in search_queries:
            print(f"\nQuery: '{query}'")
            results = await chatbot.search_memories(query, top_k=3)
            
            if results:
                print(f"  Found {len(results)} memories:")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['content'][:60]}...")
                    print(f"       Similarity: {result.get('similarity', 0):.3f}")
                    print(f"       Importance: {result.get('importance_weight', 0):.2f}")
            else:
                print("  No memories found.")
    
    finally:
        await chatbot.shutdown()


async def demo_memory_management():
    """Demo quản lý memory"""
    print("\n💾 Demo: Memory Management")
    print("=" * 50)
    
    config = MANNConfig()
    config.memory_size = 10  # Small size for demo
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Add memories until capacity is reached
        print("📝 Adding memories to test capacity management...")
        
        for i in range(15):  # More than memory_size
            memory_content = f"Memory {i+1}: This is test memory number {i+1}"
            await chatbot.process_user_input(memory_content)
            await asyncio.sleep(0.1)
        
        # Show memory statistics
        stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Memory Statistics:")
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        print(f"  Max capacity: {config.memory_size}")
        
        # Show memory contents
        print(f"\n📋 Current Memories:")
        for i, memory in enumerate(chatbot.mann_model.memory_bank, 1):
            print(f"  {i}. {memory.content[:50]}...")
            print(f"     Importance: {memory.importance_weight:.2f}")
            print(f"     Usage count: {memory.usage_count}")
    
    finally:
        await chatbot.shutdown()


async def demo_health_monitoring():
    """Demo health monitoring"""
    print("\n🏥 Demo: Health Monitoring")
    print("=" * 50)
    
    config = MANNConfig()
    config.enable_monitoring = True
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Perform some operations
        print("🔄 Performing operations to generate metrics...")
        
        for i in range(5):
            await chatbot.process_user_input(f"Test operation {i+1}")
            await asyncio.sleep(0.5)
        
        # Check health
        print("\n🏥 Health Check:")
        health = await chatbot.health_check()
        print(f"  Status: {health.get('status', 'unknown')}")
        
        checks = health.get('checks', {})
        for check_name, check_data in checks.items():
            status = check_data.get('status', 'unknown')
            status_emoji = "✅" if status == "healthy" else "❌"
            print(f"  {status_emoji} {check_name}: {status}")
        
        # Show performance stats
        if chatbot.monitor:
            perf_stats = chatbot.monitor.get_performance_stats()
            print(f"\n📈 Performance Statistics:")
            print(f"  Total queries: {perf_stats.get('total_queries', 0)}")
            print(f"  Average processing time: {perf_stats.get('avg_processing_time', 0):.3f}s")
            print(f"  Error rate: {perf_stats.get('error_rate', 0):.2%}")
            print(f"  Memory utilization: {perf_stats.get('memory_utilization', 0):.2%}")
    
    finally:
        await chatbot.shutdown()


async def demo_external_working_memory():
    """Demo External Working Memory features"""
    print("\n🧠 Demo: External Working Memory")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        print("🔬 Testing External Working Memory Operations...")
        
        # Test sequences to demonstrate learning
        test_sequences = [
            "Tôi tên là Ngọc và thích lập trình",
            "Tôi đang học về AI và Machine Learning", 
            "Tôi sống ở Hà Nội và làm việc tại ABC",
            "Tôi có sở thích đọc sách và chơi game",
            "Tôi muốn trở thành AI Engineer"
        ]
        
        print("\n📝 Processing test sequences...")
        for i, sequence in enumerate(test_sequences, 1):
            print(f"\n👤 Input {i}: {sequence}")
            
            start_time = time.time()
            response = await chatbot.process_user_input(sequence)
            processing_time = time.time() - start_time
            
            print(f"🤖 Response: {response}")
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            
            # Show memory statistics after each input
            stats = await chatbot.get_memory_statistics()
            print(f"📊 Memory stats: retrievals={stats.get('total_retrievals', 0)}, writes={stats.get('total_writes', 0)}")
            
            await asyncio.sleep(0.5)
        
        # Test memory search
        print("\n🔍 Testing Memory Search...")
        search_queries = ["tên", "lập trình", "AI", "Hà Nội", "sở thích"]
        
        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = await chatbot.search_memories(query, top_k=2)
            
            if results:
                print(f"  Found {len(results)} relevant memories:")
                for j, result in enumerate(results, 1):
                    print(f"    {j}. {result['content'][:60]}...")
                    print(f"       Similarity: {result.get('similarity', 0):.3f}")
            else:
                print("  No relevant memories found.")
        
        # Show final statistics
        final_stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Final External Working Memory Statistics:")
        print(f"  Total queries: {final_stats.get('total_queries', 0)}")
        print(f"  Total retrievals: {final_stats.get('total_retrievals', 0)}")
        print(f"  Total writes: {final_stats.get('total_writes', 0)}")
        print(f"  Memory matrix norm: {final_stats.get('memory_matrix_norm', 0):.4f}")
        print(f"  Ŵ matrix norm: {final_stats.get('W_hat_norm', 0):.4f}")
        print(f"  V̂ matrix norm: {final_stats.get('V_hat_norm', 0):.4f}")
        print(f"  Memory utilization: {final_stats.get('memory_utilization', 0):.2%}")
        
        print(f"\n✅ External Working Memory Demo Complete!")
        print(f"   - Memory Write: μ̇ᵢ = -zᵢμᵢ + cwzᵢa + zᵢŴqμᵀ")
        print(f"   - Memory Read: Mr = μz, z = softmax(μᵀq)")
        print(f"   - NN Output: uad = -Ŵᵀ(σ(V̂ᵀx̃ + b̂v) + Mr) - b̂w")
        
    finally:
        await chatbot.shutdown()


async def demo_api_integration():
    """Demo API integration"""
    print("\n🌐 Demo: API Integration")
    print("=" * 50)
    
    from standalone_mann.mann_api import MANNClient
    
    # Note: This demo assumes API server is running
    print("📡 Testing API client (requires running API server)...")
    
    try:
        async with MANNClient("http://localhost:8000") as client:
            # Health check
            print("🏥 API Health Check:")
            health = await client.health_check()
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Memory count: {health.get('memory_count', 0)}")
            
            # Add memory via API
            print("\n📝 Adding memory via API:")
            memory_id = await client.add_memory(
                content="API test memory",
                context="api_demo",
                tags=["test", "api"],
                importance_weight=1.5
            )
            print(f"  Added memory: {memory_id}")
            
            # Search via API
            print("\n🔍 Searching via API:")
            results = await client.search_memories("test", top_k=3)
            print(f"  Found {len(results)} memories")
            
            # Process query via API
            print("\n🚀 Processing query via API:")
            response = await client.process_query("Hello from API", retrieve_memories=True)
            print(f"  Response: {response.get('output', 'No output')}")
            print(f"  Processing time: {response.get('processing_time', 0):.3f}s")
            
    except Exception as e:
        print(f"❌ API demo failed: {e}")
        print("💡 Make sure API server is running: python run_api.py")


async def main():
    """Run all demos"""
    print("🎪 MANN CLI Chatbot Demo Suite")
    print("=" * 60)
    
    demos = [
        ("Basic Conversation", demo_basic_conversation),
        ("External Working Memory", demo_external_working_memory),
        ("Memory Search", demo_memory_search),
        ("Memory Management", demo_memory_management),
        ("Health Monitoring", demo_health_monitoring),
        ("API Integration", demo_api_integration)
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🎬 Starting: {demo_name}")
            await demo_func()
            print(f"✅ Completed: {demo_name}")
        except Exception as e:
            print(f"❌ Failed: {demo_name} - {e}")
        
        print("\n" + "="*60)
        await asyncio.sleep(2)  # Pause between demos
    
    print("\n🎉 All demos completed!")
    print("\n💡 To run individual demos:")
    print("  python demo.py --demo basic")
    print("  python demo.py --demo external")
    print("  python demo.py --demo search")
    print("  python demo.py --demo memory")
    print("  python demo.py --demo health")
    print("  python demo.py --demo api")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MANN CLI Chatbot Demo")
    parser.add_argument("--demo", choices=["basic", "external", "search", "memory", "health", "api"], 
                       help="Run specific demo")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_map = {
            "basic": demo_basic_conversation,
            "external": demo_external_working_memory,
            "search": demo_memory_search,
            "memory": demo_memory_management,
            "health": demo_health_monitoring,
            "api": demo_api_integration
        }
        
        print(f"🎬 Running demo: {args.demo}")
        asyncio.run(demo_map[args.demo]())
    else:
        asyncio.run(main())
