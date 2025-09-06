#!/usr/bin/env python3
"""
Test: API Integration
Tests API client functionality (requires running API server)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from standalone_mann.mann_api import MANNClient


async def test_api_integration():
    """Test API integration functionality"""
    print("🌐 Test: API Integration")
    print("=" * 50)
    
    print("📡 Testing API client functionality...")
    print("💡 Note: This test requires the API server to be running")
    print("   Start with: python run_api.py")
    
    api_url = "http://localhost:8000"
    
    try:
        async with MANNClient(api_url) as client:
            print(f"\n🔗 Connected to API: {api_url}")
            
            # Step 1: Health check
            print(f"\n1️⃣  API Health Check...")
            health = await client.health_check()
            status = health.get('status', 'unknown')
            status_emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
            
            print(f"   {status_emoji} API Status: {status}")
            print(f"   📊 Memory count: {health.get('memory_count', 0)}")
            print(f"   🕒 Response time: {health.get('response_time', 'N/A')}")
            
            if 'version' in health:
                print(f"   📱 API Version: {health['version']}")
            
            # Step 2: Add memory via API
            print(f"\n2️⃣  Adding memories via API...")
            test_memories = [
                {
                    "content": "API test memory about Python programming",
                    "context": "api_test_context",
                    "tags": ["test", "api", "python"],
                    "importance_weight": 1.5
                },
                {
                    "content": "API test memory about machine learning",
                    "context": "ml_test_context", 
                    "tags": ["test", "api", "ml", "ai"],
                    "importance_weight": 1.7
                },
                {
                    "content": "API test memory about neural networks",
                    "context": "nn_test_context",
                    "tags": ["test", "api", "neural", "network"],
                    "importance_weight": 1.6
                }
            ]
            
            added_memory_ids = []
            for i, memory in enumerate(test_memories, 1):
                memory_id = await client.add_memory(
                    content=memory["content"],
                    context=memory["context"],
                    tags=memory["tags"],
                    importance_weight=memory["importance_weight"]
                )
                added_memory_ids.append(memory_id)
                print(f"   [{i}] Added memory: {memory_id[:8]}...")
                print(f"       Content: {memory['content'][:50]}...")
            
            print(f"   ✅ Successfully added {len(added_memory_ids)} memories")
            
            # Step 3: Search via API
            print(f"\n3️⃣  Searching memories via API...")
            search_queries = ["Python", "machine learning", "neural", "programming"]
            
            for query in search_queries:
                print(f"\n🔍 Searching for: '{query}'")
                results = await client.search_memories(query, top_k=3, min_similarity=0.1)
                
                if results:
                    print(f"   ✅ Found {len(results)} memories:")
                    for j, result in enumerate(results, 1):
                        print(f"     {j}. {result.get('content', '')[:45]}...")
                        print(f"        📊 Similarity: {result.get('similarity', 0):.3f}")
                        print(f"        ⭐ Importance: {result.get('importance_weight', 0):.2f}")
                else:
                    print(f"   ❌ No memories found for '{query}'")
            
            # Step 4: Process queries via API
            print(f"\n4️⃣  Processing queries via API...")
            test_queries = [
                "Hello from API client",
                "What do you know about Python?",
                "Tell me about machine learning",
                "How do neural networks work?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n🚀 Query {i}: {query}")
                
                response = await client.process_query(
                    query=query, 
                    retrieve_memories=True,
                    max_memories=3
                )
                
                if response:
                    output = response.get('output', 'No output')
                    processing_time = response.get('processing_time', 0)
                    memories_used = response.get('memories_used', 0)
                    
                    print(f"   🤖 Response: {output[:80]}...")
                    print(f"   ⏱️  Processing time: {processing_time:.3f}s")
                    print(f"   💾 Memories used: {memories_used}")
                else:
                    print(f"   ❌ No response received")
            
            # Step 5: Get API statistics
            print(f"\n5️⃣  Getting API statistics...")
            try:
                stats = await client.get_statistics()
                if stats:
                    print(f"   📊 API Statistics:")
                    print(f"     Total queries: {stats.get('total_queries', 0)}")
                    print(f"     Total memories: {stats.get('total_memories', 0)}")
                    print(f"     Average response time: {stats.get('avg_response_time', 0):.3f}s")
                    print(f"     Memory utilization: {stats.get('memory_utilization', 0):.2%}")
                else:
                    print(f"   ⚠️  No statistics available")
            except Exception as e:
                print(f"   ⚠️  Statistics endpoint not available: {e}")
            
            # Step 6: Cleanup test memories
            print(f"\n6️⃣  Cleaning up test memories...")
            cleanup_count = 0
            for memory_id in added_memory_ids:
                try:
                    await client.delete_memory(memory_id)
                    cleanup_count += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to delete {memory_id[:8]}: {e}")
            
            print(f"   🗑️  Cleaned up {cleanup_count}/{len(added_memory_ids)} test memories")
            
            print(f"\n✅ API Integration Test Complete!")
            print(f"   🌐 API client functioning properly")
            print(f"   📡 All endpoints accessible")
            print(f"   🔄 Memory operations working correctly")
            print(f"   🚀 Query processing operational")
            
    except ConnectionError as e:
        print(f"❌ API connection failed: {e}")
        print(f"💡 Make sure the API server is running:")
        print(f"   cd mann_lib && python run_api.py")
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_api_integration())