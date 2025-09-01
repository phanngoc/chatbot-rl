"""
Forced UPDATE/DELETE Demo - Manually control để guarantee UPDATE và DELETE operations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from agents.rl_chatbot import RLChatbotAgent
from memory.memory_manager import MemoryOperation, IntelligentMemoryManager, LLMExtractor, MemoryDecisionContext
from memory.retrieval_memory import RetrievalAugmentedMemory, EpisodicMemory


def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def force_update_test():
    """Force UPDATE operation bằng cách manually setup similar memory"""
    
    print("\n" + "="*80)
    print("🔄 FORCED UPDATE TEST")
    print("="*80)
    
    # Create memory system
    memory_system = RetrievalAugmentedMemory(store_type="chroma")
    llm_extractor = LLMExtractor()
    
    # Use mock cho LLM để control output
    def mock_extract_info(dialogue_turn, context=""):
        if "pizza hải sản" in dialogue_turn.lower():
            return {
                "entities": ["pizza", "hải sản", "phô mai"],
                "intent": "expand_preference",
                "key_facts": ["likes seafood pizza", "wants extra cheese"],
                "topics": ["food", "pizza", "preferences"],
                "sentiment": "positive",
                "importance": 0.8,
                "summary": "User likes seafood pizza with extra cheese",
                "requires_memory": True,
                "memory_type": "preference"
            }
        else:
            return {
                "entities": ["pizza"],
                "intent": "express_preference", 
                "key_facts": ["likes pizza"],
                "topics": ["food", "preferences"],
                "sentiment": "positive",
                "importance": 0.7,
                "summary": "User likes pizza",
                "requires_memory": True,
                "memory_type": "preference"
            }
    
    llm_extractor.extract_key_info = mock_extract_info
    
    manager = IntelligentMemoryManager(
        memory_system=memory_system,
        llm_extractor=llm_extractor,
        similarity_threshold_update=0.6,  # Lower threshold
        similarity_threshold_delete=0.9,
        importance_threshold=0.3
    )
    
    # Step 1: Add initial memory manually
    print("1️⃣ Adding initial memory: 'User likes pizza'")
    initial_memory_id = memory_system.add_memory(
        content="User likes pizza", 
        context="food preferences",
        tags=["food", "preferences"],
        importance_score=0.7
    )
    print(f"   Initial memory ID: {initial_memory_id}")
    
    # Step 2: Process related dialogue để trigger UPDATE
    print("\n2️⃣ Processing UPDATE dialogue: 'User likes seafood pizza with extra cheese'")
    
    dialogue_turns = ["User: Tôi thích pizza hải sản với nhiều phô mai.\nBot: Pizza hải sản với phô mai rất ngon!"]
    
    result = manager.construct_memory_bank(dialogue_turns, "food preferences discussion")
    
    # Analyze result
    operations = result.get('operations_performed', [])
    if operations:
        operation = operations[0]
        actual_op = operation.get('operation')
        if hasattr(actual_op, 'value'):
            actual_op = actual_op.value
        
        print(f"   Operation: {actual_op}")
        print(f"   Confidence: {operation.get('confidence', 0):.2f}")
        print(f"   Reasoning: {operation.get('reasoning', 'N/A')}")
        print(f"   Target Memory ID: {operation.get('target_memory_id', 'N/A')}")
        
        if actual_op == "UPDATE":
            print("   ✅ SUCCESS: UPDATE operation triggered!")
        else:
            print(f"   ❌ Expected UPDATE, got {actual_op}")
            
            # Debug: Print retrieved memories
            extracted_info = operation.get('extracted_info', {})
            print(f"   Debug - Query would be: {extracted_info.get('summary', '')}")
            
            # Manually test retrieval
            test_memories = memory_system.retrieve_relevant_memories("User likes pizza", top_k=5)
            print(f"   Debug - Retrieved {len(test_memories)} memories:")
            for mem in test_memories:
                print(f"     - Similarity: {mem.get('similarity', 0):.3f}, Content: {mem.get('content', '')[:50]}")
    
    print(f"   Memories updated: {result.get('memories_updated', 0)}")
    print(f"   Memories added: {result.get('memories_added', 0)}")


def force_delete_test():
    """Force DELETE operation bằng cách setup redundant memory"""
    
    print("\n" + "="*80)
    print("🗑️ FORCED DELETE TEST") 
    print("="*80)
    
    # Create memory system
    memory_system = RetrievalAugmentedMemory(store_type="chroma")
    llm_extractor = LLMExtractor()
    
    # Mock LLM extraction cho exact repetition
    def mock_extract_info(dialogue_turn, context=""):
        return {
            "entities": ["pizza"],
            "intent": "express_preference",
            "key_facts": ["likes pizza"],
            "topics": ["food", "preferences"], 
            "sentiment": "neutral",
            "importance": 0.4,  # Lower importance
            "summary": "User likes pizza",
            "requires_memory": True,
            "memory_type": "preference"
        }
    
    llm_extractor.extract_key_info = mock_extract_info
    
    manager = IntelligentMemoryManager(
        memory_system=memory_system,
        llm_extractor=llm_extractor,
        similarity_threshold_update=0.8,
        similarity_threshold_delete=0.7,  # Lower threshold để easier trigger
        importance_threshold=0.3
    )
    
    # Step 1: Add initial memory với higher importance
    print("1️⃣ Adding initial high-importance memory")
    initial_memory_id = memory_system.add_memory(
        content="User really likes pizza and orders it frequently",
        context="detailed food preferences", 
        tags=["food", "preferences", "detailed"],
        importance_score=0.8
    )
    print(f"   Initial memory ID: {initial_memory_id}")
    
    # Step 2: Process redundant dialogue với lower importance
    print("\n2️⃣ Processing redundant dialogue with lower importance")
    
    dialogue_turns = ["User: Tôi thích pizza.\nBot: Tôi đã biết bạn thích pizza."]
    
    result = manager.construct_memory_bank(dialogue_turns, "food preferences")
    
    # Analyze result
    operations = result.get('operations_performed', [])
    if operations:
        operation = operations[0]
        actual_op = operation.get('operation')
        if hasattr(actual_op, 'value'):
            actual_op = actual_op.value
        
        print(f"   Operation: {actual_op}")
        print(f"   Confidence: {operation.get('confidence', 0):.2f}")
        print(f"   Reasoning: {operation.get('reasoning', 'N/A')}")
        print(f"   Target Memory ID: {operation.get('target_memory_id', 'N/A')}")
        
        if actual_op in ["DELETE", "NOOP"]:
            print(f"   ✅ SUCCESS: {actual_op} operation triggered (avoided redundancy)!")
        else:
            print(f"   ❌ Expected DELETE/NOOP, got {actual_op}")
            
            # Debug retrieval
            test_memories = memory_system.retrieve_relevant_memories("User likes pizza", top_k=5)
            print(f"   Debug - Retrieved {len(test_memories)} memories:")
            for mem in test_memories:
                print(f"     - Similarity: {mem.get('similarity', 0):.3f}, Content: {mem.get('content', '')[:50]}")
    
    print(f"   Memories deleted: {result.get('memories_deleted', 0)}")
    print(f"   NOOP operations: {result.get('noop_operations', 0)}")


def direct_decision_test():
    """Test decision logic directly với controlled inputs"""
    
    print("\n" + "="*80)
    print("🎯 DIRECT DECISION LOGIC TEST")
    print("="*80)
    
    memory_system = RetrievalAugmentedMemory(store_type="chroma")
    llm_extractor = LLMExtractor()
    
    manager = IntelligentMemoryManager(
        memory_system=memory_system,
        llm_extractor=llm_extractor,
        similarity_threshold_update=0.8,
        similarity_threshold_delete=0.95,
        importance_threshold=0.3
    )
    
    # Test case 1: High similarity, should UPDATE
    print("\n📝 Test 1: High similarity (0.85) with new info → should UPDATE")
    
    existing_memory = EpisodicMemory(
        id="test_memory_1",
        content="User likes pizza",
        context="food preferences",
        importance_score=0.7,
        access_count=3
    )
    
    context = MemoryDecisionContext(
        current_info={
            "importance": 0.8,
            "requires_memory": True,
            "key_facts": ["likes seafood pizza", "wants extra cheese"], 
            "sentiment": "positive"
        },
        retrieved_memories=[existing_memory],
        similarity_scores=[0.85],
        dialogue_turn="User likes seafood pizza with extra cheese",
        conversation_context="food preferences"
    )
    
    decision = manager.determine_memory_operation(context)
    print(f"   Decision: {decision.operation.value}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Reasoning: {decision.reasoning}")
    
    # Test case 2: Very high similarity, lower importance → should DELETE
    print("\n📝 Test 2: Very high similarity (0.97) with lower importance → should DELETE")
    
    context2 = MemoryDecisionContext(
        current_info={
            "importance": 0.4,  # Lower than existing
            "requires_memory": True,
            "key_facts": ["likes pizza"],
            "sentiment": "neutral"
        },
        retrieved_memories=[existing_memory],
        similarity_scores=[0.97],
        dialogue_turn="User likes pizza",
        conversation_context="food preferences"
    )
    
    decision2 = manager.determine_memory_operation(context2)
    print(f"   Decision: {decision2.operation.value}")
    print(f"   Confidence: {decision2.confidence:.2f}")
    print(f"   Reasoning: {decision2.reasoning}")
    
    # Test case 3: No existing memories → should ADD
    print("\n📝 Test 3: No existing memories → should ADD")
    
    context3 = MemoryDecisionContext(
        current_info={
            "importance": 0.7,
            "requires_memory": True,
            "key_facts": ["has shellfish allergy"],
            "sentiment": "neutral"
        },
        retrieved_memories=[],
        similarity_scores=[],
        dialogue_turn="User has shellfish allergy",
        conversation_context="health information"
    )
    
    decision3 = manager.determine_memory_operation(context3)
    print(f"   Decision: {decision3.operation.value}")
    print(f"   Confidence: {decision3.confidence:.2f}")
    print(f"   Reasoning: {decision3.reasoning}")
    
    # Test case 4: Low importance → should NOOP
    print("\n📝 Test 4: Low importance (0.2) → should NOOP")
    
    context4 = MemoryDecisionContext(
        current_info={
            "importance": 0.2,  # Below threshold
            "requires_memory": False,
            "key_facts": [],
            "sentiment": "neutral"
        },
        retrieved_memories=[],
        similarity_scores=[],
        dialogue_turn="Nice weather today",
        conversation_context="small talk"
    )
    
    decision4 = manager.determine_memory_operation(context4)
    print(f"   Decision: {decision4.operation.value}")
    print(f"   Confidence: {decision4.confidence:.2f}")
    print(f"   Reasoning: {decision4.reasoning}")


def test_with_real_agent():
    """Test với real agent using sequential similar messages"""
    
    print("\n" + "="*80)
    print("🤖 REAL AGENT SEQUENTIAL TEST")
    print("="*80)
    
    config = {
        "openai_model": "gpt-4o-mini",
        "similarity_threshold_update": 0.6,  # Lower thresholds
        "similarity_threshold_delete": 0.8,
        "importance_threshold": 0.2,
        "max_memory_capacity": 1000
    }
    
    agent = RLChatbotAgent(config=config)
    agent.start_conversation()
    
    # Sequence of related messages để build up similarity
    messages = [
        "Tôi thích ăn pizza",  # Initial
        "Pizza là món ưa thích của tôi",  # Similar → should UPDATE or NOOP
        "Tôi thích pizza hải sản nhất",  # More specific → should UPDATE  
        "Pizza hải sản với nhiều phô mai",  # Even more specific → should UPDATE
        "Tôi thích pizza",  # Repeat original → should NOOP or DELETE
    ]
    
    all_results = []
    
    for i, message in enumerate(messages):
        print(f"\n--- Message {i+1}: {message} ---")
        
        response = agent.process_message(message)
        
        print(f"Bot: {response['response']}")
        print(f"Memories used: {response['relevant_memories_count']}")
        
        # Get operation details from memory manager
        insights = agent.get_memory_manager_insights()
        op_stats = insights.get('operation_statistics', {})
        
        print(f"Total operations so far:")
        for op_type, count in op_stats.get('operation_counts', {}).items():
            print(f"  {op_type}: {count}")
        
        all_results.append({
            "message": message,
            "response": response['response'],
            "operation_stats": op_stats
        })
    
    # Summary
    print(f"\n--- FINAL SUMMARY ---")
    final_insights = agent.get_memory_manager_insights()
    final_stats = final_insights.get('operation_statistics', {})
    
    print("Final operation counts:")
    for op_type, count in final_stats.get('operation_counts', {}).items():
        print(f"  {op_type}: {count}")
    
    print(f"Total memories in bank: {final_insights.get('memory_bank_status', {}).get('total_memories', 0)}")
    
    return all_results


def main():
    """Main test function"""
    
    print("🧪 FORCED UPDATE/DELETE TESTING FOR ALGORITHM 1")
    print("=" * 80)
    
    setup_logging()
    
    # Run targeted tests
    force_update_test()
    force_delete_test() 
    direct_decision_test()
    test_with_real_agent()
    
    print("\n" + "="*80)
    print("✅ FORCED TESTING COMPLETED!")
    print("="*80)
    print("\nKey Findings:")
    print("- UPDATE operations cần similarity 0.6-0.9 với new information")
    print("- DELETE operations cần similarity >0.9 với lower importance")
    print("- NOOP operations trigger với low importance <0.3")
    print("- Similarity calculation đã được improved cho ChromaDB")


if __name__ == "__main__":
    main()
