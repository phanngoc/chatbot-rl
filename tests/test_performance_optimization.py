#!/usr/bin/env python3
"""
Test script để kiểm tra hiệu suất sau khi tối ưu hóa
"""

import sys
import os
import time
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from memory.consolidation import MemoryConsolidationSystem, KnowledgeGraph, ConsolidatedKnowledge
from datetime import datetime
import random

def generate_test_memories(num_memories: int = 1000) -> list:
    """Tạo test memories với nội dung đa dạng"""
    topics = [
        "Python programming", "Machine learning", "Deep learning", "Data science",
        "Artificial intelligence", "Neural networks", "Computer vision", "NLP",
        "Reinforcement learning", "Optimization", "Statistics", "Mathematics",
        "Software engineering", "Web development", "Mobile development", "DevOps",
        "Cloud computing", "Big data", "Database systems", "Computer networks"
    ]
    
    memories = []
    for i in range(num_memories):
        topic = random.choice(topics)
        content = f"Memory {i}: Detailed information about {topic}. "
        content += f"This includes various aspects and techniques related to {topic}. "
        content += f"Additional context and examples for {topic} implementation."
        
        memories.append({
            "id": f"mem_{i:06d}",
            "context": f"Context for {topic}",
            "content": content,
            "reward": random.uniform(0.1, 1.0),
            "timestamp": datetime.now().isoformat()
        })
    
    return memories

def test_performance_optimization():
    """Test hiệu suất sau khi tối ưu hóa"""
    print("🚀 Bắt đầu test hiệu suất tối ưu hóa...")
    
    # Test với số lượng memories khác nhau
    memory_counts = [100, 500, 1000, 2000]
    
    for count in memory_counts:
        print(f"\n📊 Testing với {count} memories...")
        
        # Tạo test memories
        test_memories = generate_test_memories(count)
        
        # Khởi tạo consolidation system
        consolidation_system = MemoryConsolidationSystem(
            consolidation_threshold=count,
            consolidation_interval_hours=1
        )
        
        # Test graph consolidation performance
        start_time = time.time()
        graph_results = consolidation_system._consolidate_via_graph(test_memories)
        graph_time = time.time() - start_time
        
        # Test similarity grouping performance
        start_time = time.time()
        similarity_results = consolidation_system._group_memories_by_similarity(test_memories)
        similarity_time = time.time() - start_time
        
        # Test knowledge graph operations
        start_time = time.time()
        graph = consolidation_system.knowledge_graph
        graph_stats = {
            "nodes": graph.graph.number_of_nodes(),
            "edges": graph.graph.number_of_edges(),
            "concepts": len(graph.concept_memories)
        }
        graph_ops_time = time.time() - start_time
        
        print(f"   📈 Graph consolidation: {graph_time:.4f}s")
        print(f"   📈 Similarity grouping: {similarity_time:.4f}s")
        print(f"   📈 Graph operations: {graph_ops_time:.4f}s")
        print(f"   📊 Results: {graph_results}")
        print(f"   📊 Similarity groups: {len(similarity_results)}")
        print(f"   📊 Graph stats: {graph_stats}")
        
        # Memory usage check
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"   💾 Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    print("\n🎯 Test hiệu suất hoàn thành!")

def test_batch_operations():
    """Test batch operations performance"""
    print("\n🔄 Testing batch operations...")
    
    # Tạo knowledge graph
    graph = KnowledgeGraph()
    
    # Test batch add knowledge
    test_knowledge = []
    for i in range(100):
        knowledge = ConsolidatedKnowledge(
            id=f"test_knowledge_{i}",
            summary=f"Test summary {i}",
            source_memories=[f"source_{i}"],
            confidence_score=0.8,
            consolidation_method="test",
            created_at=datetime.now()
        )
        test_knowledge.append((knowledge, [f"concept_{i}", f"topic_{i}"]))
    
    # Test individual vs batch operations
    print("   📊 Testing individual operations...")
    start_time = time.time()
    for knowledge, concepts in test_knowledge:
        graph.add_knowledge(knowledge, concepts)
    individual_time = time.time() - start_time
    
    print(f"   ⏱️  Individual operations: {individual_time:.4f}s")
    print(f"   📊 Graph nodes: {graph.graph.number_of_nodes()}")
    print(f"   📊 Graph edges: {graph.graph.number_of_edges()}")

if __name__ == "__main__":
    try:
        test_performance_optimization()
        test_batch_operations()
    except Exception as e:
        print(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
