"""
Test script cho OpenAI integration trong Memory Consolidation System
"""

import os
import sys
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory.consolidation import MemoryConsolidationSystem, ModelDistillation


def test_openai_integration():
    """Test OpenAI integration với sample data"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY không được set. Test với HuggingFace models...")
        use_openai = False
    else:
        print("✅ OPENAI_API_KEY found. Testing với OpenAI integration...")
        use_openai = True
    
    # Initialize consolidation system
    print(f"\n🔧 Khởi tạo MemoryConsolidationSystem (OpenAI: {use_openai})...")
    
    consolidation_system = MemoryConsolidationSystem(
        consolidation_threshold=5,  # Thấp để test
        consolidation_interval_hours=1,  # Ngắn để test
        use_openai=use_openai,
        openai_model="gpt-3.5-turbo",
        api_key=api_key
    )
    
    # Sample memories
    sample_memories = [
        {
            "id": "mem_1",
            "context": "Người dùng hỏi về machine learning",
            "content": "Machine learning là một nhánh của AI giúp máy tính học từ dữ liệu để đưa ra predictions",
            "reward": 0.9,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_2", 
            "context": "Câu hỏi về deep learning",
            "content": "Deep learning sử dụng neural networks nhiều lớp để học patterns phức tạp từ data",
            "reward": 0.85,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_3",
            "context": "Hỏi về Python programming",
            "content": "Python là ngôn ngữ lập trình dễ học, mạnh mẽ cho data science và AI",
            "reward": 0.8,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_4",
            "context": "Thảo luận về neural networks",
            "content": "Neural networks bắt chước cách hoạt động của não bộ với neurons và synapses",
            "reward": 0.75,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_5",
            "context": "Giải thích về training process",
            "content": "Training model cần data, loss function, optimizer để adjust weights qua backpropagation",
            "reward": 0.82,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_6",
            "context": "Thảo luận về overfitting",
            "content": "Overfitting xảy ra khi model học quá kỹ training data và không generalize tốt",
            "reward": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    print(f"📝 Chuẩn bị {len(sample_memories)} sample memories...")
    
    # Test consolidation
    print(f"\n🚀 Bắt đầu memory consolidation...")
    
    results = consolidation_system.consolidate_memories(
        episodic_memories=sample_memories,
        method="all"  # Test cả 3 methods
    )
    
    # Display results
    print(f"\n📊 Kết quả Consolidation:")
    print(f"   Total memories processed: {results['total_memories_processed']}")
    
    # Summarization results
    if results["summarization"]:
        print(f"\n📝 Summarization Results:")
        summ_results = results["summarization"]
        print(f"   Memories consolidated: {summ_results['memories_consolidated']}")
        print(f"   Summaries created: {summ_results['summaries_created']}")
        
        for i, summary in enumerate(summ_results['summaries'][:2]):  # Show first 2
            print(f"   Summary {i+1}: {summary['summary'][:100]}...")
    
    # Graph results
    if results["graph_integration"]:
        print(f"\n🕸️  Knowledge Graph Results:")
        graph_results = results["graph_integration"]
        print(f"   Concepts added: {graph_results['concepts_added']}")
        print(f"   Graph nodes: {graph_results['graph_nodes']}")
        print(f"   Graph edges: {graph_results['graph_edges']}")
    
    # Distillation results
    if results["distillation"]:
        print(f"\n🧠 Model Distillation Results:")
        dist_results = results["distillation"]
        print(f"   Status: {dist_results['status']}")
        print(f"   Method: {dist_results.get('method', 'unknown')}")
        
        if dist_results['status'] == 'completed':
            print(f"   Memories processed: {dist_results.get('memories_processed', 0)}")
            print(f"   Average loss: {dist_results.get('avg_loss', 0):.4f}")
            print(f"   Epochs: {dist_results.get('epochs', 0)}")
            
            # OpenAI specific stats
            if 'token_statistics' in dist_results:
                token_stats = dist_results['token_statistics']
                print(f"   Total tokens used: {token_stats.get('total_tokens', 0)}")
                print(f"   Avg token efficiency: {token_stats.get('avg_efficiency', 0):.3f}")
                print(f"   Embedding cache size: {dist_results.get('embedding_cache_size', 0)}")
    
    # Test knowledge extraction
    print(f"\n🔍 Test Knowledge Extraction...")
    test_text = "Giải thích về reinforcement learning và ứng dụng"
    
    knowledge = consolidation_system.distillation.extract_distilled_knowledge(test_text)
    
    print(f"   Input text: {test_text}")
    print(f"   Method used: {knowledge.get('method', 'unknown')}")
    
    if 'error' not in knowledge:
        print(f"   Distilled dimension: {knowledge.get('distilled_dim', 0)}")
        print(f"   Compression ratio: {knowledge.get('compression_ratio', 0):.2f}x")
        
        if 'token_analysis' in knowledge:
            token_info = knowledge['token_analysis']
            print(f"   Token count: {token_info.get('token_count', 0)}")
            print(f"   Token efficiency: {token_info.get('efficiency_score', 0):.3f}")
    else:
        print(f"   Error: {knowledge['error']}")
    
    # Test query
    print(f"\n🔎 Test Knowledge Query...")
    query_results = consolidation_system.query_consolidated_knowledge("machine learning python")
    
    print(f"   Query: 'machine learning python'")
    print(f"   Results found: {len(query_results)}")
    
    for i, result in enumerate(query_results[:2]):
        print(f"   Result {i+1}: {result['content'][:80]}...")
        print(f"     Type: {result['type']}, Confidence: {result['confidence']:.3f}")
    
    # System info
    print(f"\n🔧 System Information:")
    sys_info = consolidation_system.get_system_info()
    
    print(f"   Consolidation threshold: {sys_info['consolidation_threshold']}")
    print(f"   Total consolidated knowledge: {sys_info['total_consolidated_knowledge']}")
    print(f"   Knowledge graph: {sys_info['knowledge_graph_nodes']} nodes, {sys_info['knowledge_graph_edges']} edges")
    
    openai_info = sys_info['openai_integration']
    print(f"   OpenAI integration: {openai_info['enabled']} (available: {openai_info['available']})")
    
    if openai_info['enabled']:
        print(f"     Model: {openai_info.get('model', 'unknown')}")
        print(f"     Embedding model: {openai_info.get('embedding_model', 'unknown')}")
        print(f"     Cache size: {openai_info.get('embedding_cache_size', 0)}")
        print(f"     Method: {openai_info.get('distillation_method', 'unknown')}")
    
    # Cache management test
    if use_openai:
        print(f"\n🧹 Test Cache Management...")
        cache_cleared = consolidation_system.clear_embedding_cache()
        print(f"   Cleared {cache_cleared} cached embeddings")
    
    print(f"\n✅ Test hoàn thành!")


def test_direct_model_distillation():
    """Test ModelDistillation trực tiếp"""
    
    print(f"\n🔬 Test trực tiếp ModelDistillation...")
    
    # Test with both modes
    for use_openai in [False, True]:
        if use_openai and not os.getenv("OPENAI_API_KEY"):
            print(f"⏩ Skip OpenAI test (no API key)")
            continue
            
        print(f"\n📋 Testing ModelDistillation (OpenAI: {use_openai})...")
        
        distillation = ModelDistillation(
            use_openai=use_openai,
            openai_model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Small sample for testing
        test_memories = [
            {
                "context": "User asks about AI",
                "content": "AI stands for Artificial Intelligence",
                "reward": 0.9
            },
            {
                "context": "Question about ML", 
                "content": "Machine Learning is subset of AI",
                "reward": 0.8
            }
        ]
        
        results = distillation.distill_from_memories(
            memories=test_memories,
            num_epochs=1,  # Quick test
            batch_size=1
        )
        
        print(f"   Status: {results.get('status', 'unknown')}")
        print(f"   Method: {results.get('method', 'unknown')}")
        print(f"   Loss: {results.get('avg_loss', 0):.4f}")
        
        if use_openai and 'token_statistics' in results:
            print(f"   Tokens: {results['token_statistics'].get('total_tokens', 0)}")


if __name__ == "__main__":
    print("🧪 OpenAI Memory Consolidation Integration Test")
    print("=" * 50)
    
    try:
        test_openai_integration()
        test_direct_model_distillation()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Test script kết thúc.")
