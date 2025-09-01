"""
Targeted Demo để test UPDATE và DELETE operations trong Algorithm 1
Tạo scenarios cụ thể để trigger các operations này
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from agents.rl_chatbot import RLChatbotAgent
from memory.memory_manager import MemoryOperation, IntelligentMemoryManager, LLMExtractor
from memory.retrieval_memory import RetrievalAugmentedMemory, EpisodicMemory


def setup_logging():
    """Setup logging cho demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_update_test_scenarios() -> List[Dict[str, Any]]:
    """Tạo scenarios để test UPDATE operations"""
    
    return [
        {
            "name": "Food Preference Expansion", 
            "description": "User thêm chi tiết về sở thích đã có",
            "initial_dialogue": "User: Tôi thích pizza.\nBot: Loại pizza nào bạn thích?",
            "update_dialogue": "User: Tôi thích pizza hải sản với nhiều phô mai.\nBot: Pizza hải sản với phô mai rất ngon!",
            "expected_operation": "UPDATE",
            "reasoning": "Thêm details vào existing pizza preference"
        },
        {
            "name": "Allergy Information Addition",
            "description": "User cung cấp thêm thông tin về dị ứng",
            "initial_dialogue": "User: Tôi bị dị ứng tôm.\nBot: Tôi sẽ nhớ để tránh gợi ý món có tôm.",
            "update_dialogue": "User: Tôi cũng dị ứng cua và tất cả hải sản.\nBot: Tôi hiểu, sẽ tránh tất cả hải sản.",
            "expected_operation": "UPDATE", 
            "reasoning": "Mở rộng thông tin dị ứng từ tôm sang tất cả hải sản"
        },
        {
            "name": "Dietary Preference Change",
            "description": "User thay đổi thông tin về chế độ ăn",
            "initial_dialogue": "User: Tôi đang ăn kiêng keto.\nBot: Tôi sẽ gợi ý món low-carb cho bạn.",
            "update_dialogue": "User: Tôi đã ngừng keto, giờ ăn bình thường.\nBot: Được rồi, tôi sẽ cập nhật thông tin.",
            "expected_operation": "UPDATE",
            "reasoning": "Thay đổi dietary status từ keto sang normal"
        }
    ]


def create_delete_test_scenarios() -> List[Dict[str, Any]]:
    """Tạo scenarios để test DELETE operations"""
    
    return [
        {
            "name": "Exact Repetition",
            "description": "User lặp lại thông tin hoàn toàn giống nhau",
            "initial_dialogue": "User: Tôi thích pizza.\nBot: Loại pizza nào bạn thích?",
            "repeat_dialogue": "User: Tôi thích pizza.\nBot: Vâng, tôi đã biết bạn thích pizza.",
            "expected_operation": "DELETE hoặc NOOP",
            "reasoning": "Thông tin hoàn toàn trùng lặp"
        },
        {
            "name": "Less Important Information",
            "description": "User cung cấp thông tin ít quan trọng hơn thông tin đã có",
            "initial_dialogue": "User: Tôi bị dị ứng nghiêm trọng với tôm, có thể gây sốc phản vệ.\nBot: Tôi sẽ cực kỳ cẩn thận với hải sản.",
            "repeat_dialogue": "User: Tôi không ăn tôm.\nBot: Vâng, tôi đã ghi nhận điều này.",
            "expected_operation": "DELETE hoặc NOOP",
            "reasoning": "Information mới ít chi tiết và quan trọng hơn existing"
        },
        {
            "name": "Vague Restatement",
            "description": "User nói lại một cách mơ hồ thông tin đã rõ ràng",
            "initial_dialogue": "User: Tôi thích pizza Margherita với cà chua tươi và basil.\nBot: Pizza Margherita là lựa chọn tuyệt vời!",
            "repeat_dialogue": "User: Tôi thích pizza.\nBot: Đúng, bạn thích pizza Margherita.",
            "expected_operation": "DELETE hoặc NOOP", 
            "reasoning": "Thông tin mới quá general so với existing detailed info"
        }
    ]


def create_noop_test_scenarios() -> List[Dict[str, Any]]:
    """Tạo scenarios để test NOOP operations"""
    
    return [
        {
            "name": "Low Importance Small Talk",
            "description": "Conversation không có giá trị memory",
            "dialogue": "User: Trời hôm nay đẹp quá.\nBot: Vâng, thời tiết rất tốt.",
            "expected_operation": "NOOP",
            "reasoning": "Small talk, low importance"
        },
        {
            "name": "Greeting Exchange",
            "description": "Chào hỏi thông thường",
            "dialogue": "User: Xin chào!\nBot: Chào bạn! Tôi có thể giúp gì cho bạn?",
            "expected_operation": "NOOP",
            "reasoning": "Greeting, không cần memory"
        }
    ]


def manually_add_initial_memory(agent: RLChatbotAgent, content: str, context: str, importance: float = 0.8) -> str:
    """Manually add initial memory để setup test scenarios"""
    
    memory_id = agent.retrieval_memory.add_memory(
        content=content,
        context=context,
        tags=["test", "initial"],
        importance_score=importance,
        metadata={
            "test_setup": True,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    return memory_id


def test_update_operations():
    """Test UPDATE operations với specific scenarios"""
    
    print("\n" + "="*80)
    print("🔄 TESTING UPDATE OPERATIONS")
    print("="*80)
    
    config = {
        "openai_model": "gpt-4o-mini",
        "similarity_threshold_update": 0.7,  # Lower threshold để easier trigger UPDATE
        "similarity_threshold_delete": 0.9,
        "importance_threshold": 0.2,
        "max_memory_capacity": 1000
    }
    
    agent = RLChatbotAgent(config=config)
    agent.start_conversation()
    
    update_scenarios = create_update_test_scenarios()
    
    for i, scenario in enumerate(update_scenarios):
        print(f"\n--- UPDATE TEST {i+1}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Expected: {scenario['expected_operation']}")
        print(f"Reasoning: {scenario['reasoning']}")
        
        # Step 1: Add initial memory
        print(f"\n1️⃣ Initial: {scenario['initial_dialogue']}")
        initial_result = agent.process_dialogue_batch([scenario['initial_dialogue']])
        initial_memories = initial_result.get('memories_added', 0)
        print(f"   Added {initial_memories} initial memories")
        
        # Step 2: Process update dialogue
        print(f"\n2️⃣ Update: {scenario['update_dialogue']}")
        update_result = agent.process_dialogue_batch([scenario['update_dialogue']])
        
        # Analyze results
        operations = update_result.get('operations_performed', [])
        if operations:
            operation = operations[0]
            actual_op = operation.get('operation')
            if hasattr(actual_op, 'value'):
                actual_op = actual_op.value
            
            confidence = operation.get('confidence', 0)
            reasoning = operation.get('reasoning', 'N/A')
            
            print(f"   Actual Operation: {actual_op}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Reasoning: {reasoning}")
            
            # Check if it matches expectation
            if actual_op == scenario['expected_operation']:
                print(f"   ✅ SUCCESS: Got expected {scenario['expected_operation']} operation")
            else:
                print(f"   ❌ UNEXPECTED: Expected {scenario['expected_operation']}, got {actual_op}")
        
        print(f"   Updates: {update_result.get('memories_updated', 0)}")
        print(f"   Adds: {update_result.get('memories_added', 0)}")
        print(f"   NOOPs: {update_result.get('noop_operations', 0)}")


def test_delete_operations():
    """Test DELETE operations với specific scenarios"""
    
    print("\n" + "="*80)
    print("🗑️ TESTING DELETE OPERATIONS")
    print("="*80)
    
    config = {
        "openai_model": "gpt-4o-mini", 
        "similarity_threshold_update": 0.8,
        "similarity_threshold_delete": 0.85,  # Lower threshold để easier trigger DELETE
        "importance_threshold": 0.3,
        "max_memory_capacity": 1000
    }
    
    agent = RLChatbotAgent(config=config)
    agent.start_conversation()
    
    delete_scenarios = create_delete_test_scenarios()
    
    for i, scenario in enumerate(delete_scenarios):
        print(f"\n--- DELETE TEST {i+1}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Expected: {scenario['expected_operation']}")
        print(f"Reasoning: {scenario['reasoning']}")
        
        # Step 1: Add initial memory
        print(f"\n1️⃣ Initial: {scenario['initial_dialogue']}")
        initial_result = agent.process_dialogue_batch([scenario['initial_dialogue']])
        initial_memories = initial_result.get('memories_added', 0)
        print(f"   Added {initial_memories} initial memories")
        
        # Step 2: Process repeat dialogue
        print(f"\n2️⃣ Repeat: {scenario['repeat_dialogue']}")
        repeat_result = agent.process_dialogue_batch([scenario['repeat_dialogue']])
        
        # Analyze results
        operations = repeat_result.get('operations_performed', [])
        if operations:
            operation = operations[0]
            actual_op = operation.get('operation')
            if hasattr(actual_op, 'value'):
                actual_op = actual_op.value
            
            confidence = operation.get('confidence', 0)
            reasoning = operation.get('reasoning', 'N/A')
            
            print(f"   Actual Operation: {actual_op}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Reasoning: {reasoning}")
            
            # Check if it matches expectation
            expected_ops = scenario['expected_operation'].split(' hoặc ')
            if actual_op in expected_ops:
                print(f"   ✅ SUCCESS: Got expected operation ({actual_op})")
            else:
                print(f"   ❌ UNEXPECTED: Expected {scenario['expected_operation']}, got {actual_op}")
        
        print(f"   Deletes: {repeat_result.get('memories_deleted', 0)}")
        print(f"   Adds: {repeat_result.get('memories_added', 0)}")
        print(f"   NOOPs: {repeat_result.get('noop_operations', 0)}")


def test_noop_operations():
    """Test NOOP operations với specific scenarios"""
    
    print("\n" + "="*80)
    print("⏸️ TESTING NOOP OPERATIONS")
    print("="*80)
    
    config = {
        "openai_model": "gpt-4o-mini",
        "importance_threshold": 0.4,  # Higher threshold để easier trigger NOOP
        "max_memory_capacity": 1000
    }
    
    agent = RLChatbotAgent(config=config)
    agent.start_conversation()
    
    noop_scenarios = create_noop_test_scenarios()
    
    for i, scenario in enumerate(noop_scenarios):
        print(f"\n--- NOOP TEST {i+1}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Expected: {scenario['expected_operation']}")
        print(f"Reasoning: {scenario['reasoning']}")
        
        print(f"\n💬 Dialogue: {scenario['dialogue']}")
        result = agent.process_dialogue_batch([scenario['dialogue']])
        
        # Analyze results
        operations = result.get('operations_performed', [])
        if operations:
            operation = operations[0]
            actual_op = operation.get('operation')
            if hasattr(actual_op, 'value'):
                actual_op = actual_op.value
            
            confidence = operation.get('confidence', 0)
            reasoning = operation.get('reasoning', 'N/A')
            extracted_info = operation.get('extracted_info', {})
            importance = extracted_info.get('importance', 0)
            
            print(f"   Actual Operation: {actual_op}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Importance: {importance:.2f}")
            print(f"   Reasoning: {reasoning}")
            
            # Check if it matches expectation
            if actual_op == scenario['expected_operation']:
                print(f"   ✅ SUCCESS: Got expected {scenario['expected_operation']} operation")
            else:
                print(f"   ❌ UNEXPECTED: Expected {scenario['expected_operation']}, got {actual_op}")
        
        print(f"   NOOPs: {result.get('noop_operations', 0)}")
        print(f"   Adds: {result.get('memories_added', 0)}")


def test_similarity_based_decisions():
    """Test decision logic dựa trên similarity scores cụ thể"""
    
    print("\n" + "="*80)
    print("📊 TESTING SIMILARITY-BASED DECISIONS")
    print("="*80)
    
    # Create memory manager để test directly
    memory_system = RetrievalAugmentedMemory(store_type="chroma")
    llm_extractor = LLMExtractor()
    
    manager = IntelligentMemoryManager(
        memory_system=memory_system,
        llm_extractor=llm_extractor,
        similarity_threshold_update=0.8,
        similarity_threshold_delete=0.95,
        importance_threshold=0.3
    )
    
    # Test cases với different similarity scores
    test_cases = [
        {
            "similarity": 0.5,
            "importance": 0.8,
            "expected": "ADD",
            "description": "Low similarity, high importance"
        },
        {
            "similarity": 0.85,
            "importance": 0.7,
            "expected": "UPDATE", 
            "description": "High similarity, good importance"
        },
        {
            "similarity": 0.97,
            "importance": 0.4,
            "expected": "DELETE",
            "description": "Very high similarity, lower importance"
        },
        {
            "similarity": 0.85,
            "importance": 0.1,
            "expected": "NOOP",
            "description": "High similarity, very low importance"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- SIMILARITY TEST {i+1}: {test_case['description']} ---")
        print(f"Similarity: {test_case['similarity']:.2f}")
        print(f"Importance: {test_case['importance']:.2f}")
        print(f"Expected: {test_case['expected']}")
        
        # Create mock memory context
        from memory.memory_manager import MemoryDecisionContext
        from memory.retrieval_memory import EpisodicMemory
        
        # Create existing memory để test against
        existing_memory = EpisodicMemory(
            id="test_memory",
            content="Test existing content",
            context="Test context",
            importance_score=0.6,
            access_count=5
        )
        
        context = MemoryDecisionContext(
            current_info={
                "importance": test_case['importance'],
                "requires_memory": True,
                "key_facts": ["test fact"],
                "sentiment": "neutral"
            },
            retrieved_memories=[existing_memory] if test_case['similarity'] > 0.1 else [],
            similarity_scores=[test_case['similarity']] if test_case['similarity'] > 0.1 else [],
            dialogue_turn="Test dialogue",
            conversation_context="Test context"
        )
        
        # Get decision
        result = manager.determine_memory_operation(context)
        
        actual_op = result.operation.value if hasattr(result.operation, 'value') else str(result.operation)
        
        print(f"Actual Operation: {actual_op}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        
        if actual_op == test_case['expected']:
            print(f"✅ SUCCESS: Correct decision")
        else:
            print(f"❌ FAILURE: Expected {test_case['expected']}, got {actual_op}")


def main():
    """Main test function"""
    
    print("🧪 COMPREHENSIVE UPDATE/DELETE/NOOP TESTING FOR ALGORITHM 1")
    print("=" * 80)
    
    setup_logging()
    
    # Run all tests
    test_update_operations()
    test_delete_operations() 
    test_noop_operations()
    test_similarity_based_decisions()
    
    print("\n" + "="*80)
    print("🎯 TESTING COMPLETED!")
    print("="*80)
    print("\nNotes:")
    print("- UPDATE operations require similar content với new information")
    print("- DELETE operations require very high similarity với lower importance")
    print("- NOOP operations trigger with low importance hoặc sufficient existing memory")
    print("- Similarity thresholds có thể cần điều chỉnh để optimize performance")


if __name__ == "__main__":
    main()
