#!/usr/bin/env python3
"""
Test: Memory Search
Tests memory search and retrieval functionality using OpenAI embeddings
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def test_memory_search():
    """Test memory search functionality"""
    print("🔍 Test: Memory Search")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Add test memories
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
        
        print(f"📝 Adding {len(test_memories)} test memories...")
        for i, memory in enumerate(test_memories, 1):
            await chatbot.process_user_input(memory)
            print(f"  [{i}] Added: {memory[:50]}...")
            await asyncio.sleep(0.3)
        
        # Test search queries
        search_queries = [
            "tên",
            "lập trình",
            "AI",
            "Hà Nội", 
            "mèo",
            "sở thích",
            "công việc",
            "học tập"
        ]
        
        print(f"\n🔍 Testing search with {len(search_queries)} queries...")
        
        for query in search_queries:
            print(f"\n🔎 Query: '{query}'")
            results = await chatbot.search_memories(query, top_k=3, min_similarity=0.1)
            
            if results:
                print(f"  ✅ Found {len(results)} memories:")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['content'][:60]}...")
                    print(f"       📊 Similarity: {result.get('similarity', 0):.3f}")
                    print(f"       ⭐ Importance: {result.get('importance_weight', 0):.2f}")
            else:
                print("  ❌ No memories found")
        
        # Test semantic search
        print(f"\n🧠 Testing semantic search...")
        semantic_queries = [
            "nghề nghiệp của tôi",
            "động vật nuôi",
            "ngôn ngữ lập trình yêu thích",
            "nơi sinh sống"
        ]
        
        for query in semantic_queries:
            print(f"\n🔍 Semantic query: '{query}'")
            results = await chatbot.search_memories(query, top_k=2, min_similarity=0.2)
            
            if results:
                print(f"  ✅ Found {len(results)} semantic matches:")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['content']}")
                    print(f"       📊 Similarity: {result.get('similarity', 0):.3f}")
            else:
                print("  ❌ No semantic matches found")
        
        # Final statistics
        stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Search Test Statistics:")
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Total searches: {len(search_queries) + len(semantic_queries)}")
        print(f"  Total retrievals: {stats.get('total_retrievals', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        
        print("\n✅ Memory Search Test Complete!")
        
    finally:
        await chatbot.shutdown()


if __name__ == "__main__":
    asyncio.run(test_memory_search())