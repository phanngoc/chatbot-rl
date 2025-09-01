"""
Demo script để test Algorithm 1: Memory Bank Construction via Memory Manager
Minh họa intelligent memory operations: ADD, UPDATE, DELETE, NOOP
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from agents.rl_chatbot import RLChatbotAgent
from memory.memory_manager import MemoryOperation


def setup_logging():
    """Setup logging cho demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('algorithm_1_demo.log')
        ]
    )


def create_sample_dialogue_turns() -> List[str]:
    """Tạo sample dialogue turns để test"""
    
    return [
        # Initial conversation about preferences
        "User: Tôi thích ăn pizza.\nBot: Thật tuyệt! Bạn thích loại pizza nào nhất?",
        
        # More details about pizza preference
        "User: Tôi thích pizza hải sản với nhiều phô mai.\nBot: Pizza hải sản với phô mai nghe rất ngon! Bạn thường đặt ở đâu?",
        
        # Location information
        "User: Tôi thường đặt ở Pizza Hut gần nhà.\nBot: Pizza Hut có nhiều lựa chọn tốt. Bạn có thường xuyên đặt online không?",
        
        # Contradictory information (should trigger UPDATE)
        "User: Thực ra tôi không thích pizza lắm, tôi thích bánh mì hơn.\nBot: À, vậy bánh mì là món ưa thích của bạn. Bánh mì Việt Nam hay bánh mì Tây?",
        
        # Redundant information (should trigger NOOP or DELETE)
        "User: Tôi thích bánh mì.\nBot: Vâng, tôi đã ghi nhớ bạn thích bánh mì.",
        
        # New important information (should trigger ADD)
        "User: Tôi bị dị ứng với tôm, cần tránh trong thức ăn.\nBot: Cảm ơn bạn đã cho tôi biết. Tôi sẽ nhớ tránh gợi ý món có tôm cho bạn.",
        
        # Related but additional information (should trigger UPDATE)
        "User: Ngoài tôm, tôi cũng không ăn được cua và các loại hải sản khác.\nBot: Tôi hiểu rồi, bạn có dị ứng với hải sản nói chung. Tôi sẽ cẩn thận khi gợi ý món ăn.",
        
        # Low importance information (should trigger NOOP)
        "User: Hôm nay trời đẹp quá.\nBot: Vâng, thời tiết hôm nay thật tuyệt.",
        
        # Specific dietary preference (should trigger ADD)
        "User: Tôi đang ăn kiêng keto, cần hạn chế carbs.\nBot: Tôi sẽ gợi ý những món ăn phù hợp với chế độ keto cho bạn.",
        
        # Conflicting with previous info (should trigger UPDATE)
        "User: Thực ra tôi đã ngừng ăn kiêng keto rồi, giờ tôi ăn bình thường.\nBot: Được rồi, tôi sẽ cập nhật thông tin về chế độ ăn của bạn."
    ]


def print_memory_operation_analysis(operations: List[Dict[str, Any]]):
    """In phân tích chi tiết về các memory operations"""
    
    print("\n" + "="*80)
    print("PHÂN TÍCH CHI TIẾT CÁC MEMORY OPERATIONS")
    print("="*80)
    
    operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
    
    for i, op in enumerate(operations):
        operation = op.get("operation", MemoryOperation.NOOP)
        if hasattr(operation, 'value'):
            operation = operation.value
        
        operation_counts[str(operation)] += 1
        
        print(f"\n--- Turn {i+1}: {operation} ---")
        print(f"Confidence: {op.get('confidence', 0):.2f}")
        print(f"Reasoning: {op.get('reasoning', 'N/A')}")
        
        if op.get('target_memory_id'):
            print(f"Target Memory ID: {op.get('target_memory_id')}")
        
        extracted_info = op.get('extracted_info', {})
        if extracted_info:
            print(f"Importance: {extracted_info.get('importance', 0):.2f}")
            print(f"Sentiment: {extracted_info.get('sentiment', 'N/A')}")
            print(f"Topics: {extracted_info.get('topics', [])}")
            print(f"Key Facts: {extracted_info.get('key_facts', [])}")
        
        execution_result = op.get('execution_result', {})
        if execution_result:
            print(f"Execution Success: {execution_result.get('success', False)}")
            print(f"Message: {execution_result.get('message', 'N/A')}")
    
    print(f"\n--- OPERATION SUMMARY ---")
    total_ops = sum(operation_counts.values())
    for op_type, count in operation_counts.items():
        percentage = (count / total_ops * 100) if total_ops > 0 else 0
        print(f"{op_type}: {count} operations ({percentage:.1f}%)")


def demonstrate_algorithm_1():
    """Main demo function"""
    
    print("🤖 ALGORITHM 1 DEMONSTRATION: Memory Bank Construction via Memory Manager")
    print("=" * 80)
    
    # Setup agent với intelligent memory manager
    config = {
        "openai_model": "gpt-4o-mini",
        "max_tokens": 150,
        "temperature": 0.7,
        "similarity_threshold_update": 0.8,
        "similarity_threshold_delete": 0.95,
        "importance_threshold": 0.3,
        "max_memory_capacity": 1000,
        "llm_model": "gpt-4o-mini"
    }
    
    agent = RLChatbotAgent(
        openai_model="gpt-4o-mini",
        config=config
    )
    
    # Start conversation
    conversation_id = agent.start_conversation()
    print(f"Started conversation: {conversation_id}")
    
    # Get sample dialogue turns
    dialogue_turns = create_sample_dialogue_turns()
    print(f"\nTesting với {len(dialogue_turns)} dialogue turns...")
    
    # Process dialogue batch sử dụng Algorithm 1
    print("\n📝 PROCESSING DIALOGUE TURNS WITH ALGORITHM 1...")
    print("-" * 60)
    
    batch_results = agent.process_dialogue_batch(
        dialogue_turns=dialogue_turns,
        context="Demo conversation về food preferences và dietary restrictions"
    )
    
    # Display results
    print(f"\n✅ BATCH PROCESSING COMPLETED")
    print(f"Total turns processed: {batch_results.get('total_turns_processed', 0)}")
    print(f"Memories added: {batch_results.get('memories_added', 0)}")
    print(f"Memories updated: {batch_results.get('memories_updated', 0)}")
    print(f"Memories deleted: {batch_results.get('memories_deleted', 0)}")
    print(f"NOOP operations: {batch_results.get('noop_operations', 0)}")
    print(f"Processing errors: {batch_results.get('processing_errors', 0)}")
    
    # Detailed operation analysis
    operations = batch_results.get('operations_performed', [])
    if operations:
        print_memory_operation_analysis(operations)
    
    # Memory Manager insights
    print("\n" + "="*80)
    print("MEMORY MANAGER INSIGHTS")
    print("="*80)
    
    insights = agent.get_memory_manager_insights()
    
    op_stats = insights.get('operation_statistics', {})
    print(f"\nOperation Statistics:")
    for op_type, count in op_stats.get('operation_counts', {}).items():
        percentage = op_stats.get('operation_percentages', {}).get(op_type, 0)
        print(f"  {op_type}: {count} ({percentage:.1f}%)")
    
    config_info = insights.get('configuration', {})
    print(f"\nConfiguration:")
    print(f"  Similarity threshold (UPDATE): {config_info.get('similarity_threshold_update', 'N/A')}")
    print(f"  Similarity threshold (DELETE): {config_info.get('similarity_threshold_delete', 'N/A')}")
    print(f"  Importance threshold: {config_info.get('importance_threshold', 'N/A')}")
    print(f"  Max memory capacity: {config_info.get('max_memory_capacity', 'N/A')}")
    
    memory_status = insights.get('memory_bank_status', {})
    print(f"\nMemory Bank Status:")
    print(f"  Total memories: {memory_status.get('total_memories', 0)}")
    print(f"  Near capacity: {memory_status.get('is_near_capacity', False)}")
    
    # Test interactive conversation
    print("\n" + "="*80)
    print("INTERACTIVE CONVERSATION TEST")
    print("="*80)
    
    test_messages = [
        "Tôi muốn đặt pizza hôm nay",
        "Bạn có nhớ tôi thích gì không?",
        "Tôi có dị ứng gì cần lưu ý không?"
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        response = agent.process_message(msg)
        print(f"🤖 Bot: {response['response']}")
        print(f"   Memories used: {response['relevant_memories_count']}")
        print(f"   Response time: {response['response_time_ms']:.1f}ms")
    
    # Final system status
    print("\n" + "="*80)
    print("FINAL SYSTEM STATUS")
    print("="*80)
    
    status = agent.get_system_status()
    print(f"Model: {status['model_info']['model_name']}")
    print(f"Total interactions: {status['performance_metrics']['total_interactions']}")
    print(f"Memory consolidation: {status['system_health']['memory_consolidation_status']}")
    print(f"Intelligent Memory Manager: {status['system_health']['intelligent_memory_manager']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"algorithm_1_results_{timestamp}.json"
    
    full_results = {
        "demo_info": {
            "timestamp": timestamp,
            "dialogue_turns_count": len(dialogue_turns),
            "conversation_id": conversation_id
        },
        "batch_processing_results": batch_results,
        "memory_manager_insights": insights,
        "system_status": status,
        "configuration": config
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    print("\n🎉 ALGORITHM 1 DEMONSTRATION COMPLETED!")


if __name__ == "__main__":
    setup_logging()
    demonstrate_algorithm_1()
