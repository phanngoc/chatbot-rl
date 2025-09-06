#!/usr/bin/env python3
"""
Test: External Working Memory
Tests the external working memory operations and equations
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def test_external_working_memory():
    """Test external working memory functionality"""
    print("🧠 Test: External Working Memory")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        print("🔬 Testing External Working Memory Operations...")
        print("   - Memory Write: μ̇ᵢ = -zᵢμᵢ + cwzᵢa + zᵢŴqμᵀ")
        print("   - Memory Read: Mr = μz, z = softmax(μᵀq)")
        print("   - NN Output: uad = -Ŵᵀ(σ(V̂ᵀx̃ + b̂v) + Mr) - b̂w")
        
        # Test sequences to demonstrate memory operations
        test_sequences = [
            "Tôi tên là Ngọc và thích lập trình",
            "Tôi đang học về AI và Machine Learning", 
            "Tôi sống ở Hà Nội và làm việc tại ABC",
            "Tôi có sở thích đọc sách và chơi game",
            "Tôi muốn trở thành AI Engineer",
            "Tôi đang phát triển chatbot thông minh",
            "Tôi sử dụng PyTorch cho deep learning",
            "Tôi quan tâm đến reinforcement learning"
        ]
        
        print(f"\n📝 Processing {len(test_sequences)} test sequences...")
        
        for i, sequence in enumerate(test_sequences, 1):
            print(f"\n👤 Input {i}: {sequence}")
            
            start_time = time.time()
            response = await chatbot.process_user_input(sequence)
            processing_time = time.time() - start_time
            
            print(f"🤖 Response: {response}")
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            
            # Show memory operations after each input
            stats = await chatbot.get_memory_statistics()
            print(f"📊 Memory ops: retrievals={stats.get('total_retrievals', 0)}, writes={stats.get('total_writes', 0)}")
            
            await asyncio.sleep(0.3)
        
        # Test memory read operations
        print(f"\n🔍 Testing Memory Read Operations...")
        search_queries = [
            "tên", "lập trình", "AI", "Hà Nội", 
            "sở thích", "công việc", "PyTorch", "reinforcement"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Reading memories for: '{query}'")
            results = await chatbot.search_memories(query, top_k=2)
            
            if results:
                print(f"  📖 Read {len(results)} relevant memories:")
                for j, result in enumerate(results, 1):
                    print(f"    {j}. {result['content'][:50]}...")
                    print(f"       📊 Similarity: {result.get('similarity', 0):.3f}")
                    print(f"       🔄 Usage: {result.get('usage_count', 0)}")
            else:
                print("  ❌ No relevant memories found")
        
        # Test memory write/update operations
        print(f"\n✏️  Testing Memory Write Operations...")
        update_sequences = [
            "Tôi đã học thêm về Transformer architecture",
            "Tôi vừa hoàn thành project chatbot đầu tiên",
            "Tôi quan tâm đến việc deploy AI model"
        ]
        
        for sequence in update_sequences:
            print(f"\n📝 Writing: {sequence}")
            await chatbot.process_user_input(sequence)
        
        # Final statistics showing all memory operations
        final_stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Final External Working Memory Statistics:")
        print(f"  🧮 Total queries: {final_stats.get('total_queries', 0)}")
        print(f"  📖 Total reads (retrievals): {final_stats.get('total_retrievals', 0)}")
        print(f"  ✏️  Total writes: {final_stats.get('total_writes', 0)}")
        print(f"  🧠 Memory matrix norm: {final_stats.get('memory_matrix_norm', 0):.4f}")
        print(f"  ⚖️  Ŵ matrix norm: {final_stats.get('W_hat_norm', 0):.4f}")
        print(f"  📊 V̂ matrix norm: {final_stats.get('V_hat_norm', 0):.4f}")
        print(f"  📈 Memory utilization: {final_stats.get('memory_utilization', 0):.2%}")
        
        # Show memory bank state
        if hasattr(chatbot, 'mann_model') and chatbot.mann_model.memory_bank:
            print(f"\n💾 Memory Bank State:")
            print(f"  📝 Total memories: {len(chatbot.mann_model.memory_bank)}")
            print(f"  🔄 Average usage: {sum(m.usage_count for m in chatbot.mann_model.memory_bank) / len(chatbot.mann_model.memory_bank):.1f}")
            print(f"  ⭐ Average importance: {sum(m.importance_weight for m in chatbot.mann_model.memory_bank) / len(chatbot.mann_model.memory_bank):.2f}")
        
        print(f"\n✅ External Working Memory Test Complete!")
        print(f"   🧮 Demonstrated memory read, write, and attention operations")
        
    finally:
        await chatbot.shutdown()


if __name__ == "__main__":
    asyncio.run(test_external_working_memory())