#!/usr/bin/env python3
"""
Test: Memory Management
Tests memory capacity management and cleanup
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def test_memory_management():
    """Test memory management functionality"""
    print("💾 Test: Memory Management")
    print("=" * 50)
    
    config = MANNConfig()
    config.memory_size = 8  # Small size for testing
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        print(f"📏 Testing with memory limit: {config.memory_size}")
        
        # Add memories beyond capacity
        test_memories = [
            f"Memory {i+1}: Test content about topic {i+1} with importance {i+1}" 
            for i in range(15)  # More than memory_size
        ]
        
        print(f"\n📝 Adding {len(test_memories)} memories (exceeds capacity)...")
        
        for i, memory_content in enumerate(test_memories):
            await chatbot.process_user_input(memory_content)
            
            # Show progress every few additions
            if (i + 1) % 3 == 0:
                stats = await chatbot.get_memory_statistics()
                current_count = stats.get('total_memories', 0)
                utilization = stats.get('memory_utilization', 0)
                print(f"  [{i+1:2d}] Added memory | Current: {current_count}, Utilization: {utilization:.1%}")
            
            await asyncio.sleep(0.1)
        
        # Final memory statistics
        final_stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Final Memory Statistics:")
        print(f"  Total memories: {final_stats.get('total_memories', 0)}")
        print(f"  Memory utilization: {final_stats.get('memory_utilization', 0):.2%}")
        print(f"  Max capacity: {config.memory_size}")
        print(f"  Total writes: {final_stats.get('total_writes', 0)}")
        print(f"  Total retrievals: {final_stats.get('total_retrievals', 0)}")
        
        # Show current memory contents
        print(f"\n📋 Current Memory Bank Contents:")
        if hasattr(chatbot, 'mann_model') and chatbot.mann_model.memory_bank:
            for i, memory in enumerate(chatbot.mann_model.memory_bank, 1):
                print(f"  [{i}] {memory.content[:60]}...")
                print(f"      ⭐ Importance: {memory.importance_weight:.2f}")
                print(f"      🔄 Usage: {memory.usage_count}")
                print(f"      📅 Time: {memory.timestamp.strftime('%H:%M:%S')}")
        else:
            print("  No memories found")
        
        # Test memory retrieval after cleanup
        print(f"\n🔍 Testing memory retrieval after capacity management...")
        test_queries = ["topic 1", "topic 5", "topic 10", "topic 15"]
        
        for query in test_queries:
            results = await chatbot.search_memories(query, top_k=2)
            found_count = len(results) if results else 0
            print(f"  Query '{query}': {found_count} matches")
            if results:
                for result in results:
                    print(f"    - {result['content'][:50]}... (sim: {result.get('similarity', 0):.3f})")
        
        print("\n✅ Memory Management Test Complete!")
        print(f"   📈 Successfully managed {len(test_memories)} memories with limit of {config.memory_size}")
        
    finally:
        await chatbot.shutdown()


if __name__ == "__main__":
    asyncio.run(test_memory_management())