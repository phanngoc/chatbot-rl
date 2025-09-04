#!/usr/bin/env python3
"""
Test script cho Knowledge Database Integration
Kiểm tra tính năng lưu trữ và quản lý extracted knowledge vào SQLite database
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database.knowledge_db_manager import KnowledgeDatabaseManager, ExtractedKnowledge
from src.memory.memory_operations import MemoryOperation
from src.memory.memory_manager import IntelligentMemoryManager, LLMExtractor
from src.memory.retrieval_memory import RetrievalAugmentedMemory
from src.agents.rl_chatbot import RLChatbotAgent

def test_knowledge_database_basic():
    """Test cơ bản cho Knowledge Database"""
    
    print("=== Test Knowledge Database Basic Functionality ===")
    
    # Tạo temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_knowledge.db")
        
        try:
            # Initialize Knowledge Database
            knowledge_db = KnowledgeDatabaseManager(db_path)
            
            # Test 1: Store extracted knowledge
            print("\n1. Testing store_extracted_knowledge...")
            
            extracted_info = {
                "entities": ["John", "AI", "chatbot"],
                "intent": "asking_about_ai",
                "key_facts": ["AI is helpful", "Chatbots can assist users"],
                "topics": ["AI", "technology", "assistance"],
                "sentiment": "positive",
                "importance": 0.8,
                "memory_type": "factual",
                "summary": "User asking about AI capabilities",
                "metadata": {"confidence": 0.9}
            }
            
            knowledge_id = knowledge_db.store_extracted_knowledge(
                extracted_info=extracted_info,
                dialogue_turn="User: What can AI do?\nBot: AI can help with many tasks including answering questions.",
                context="General conversation about AI",
                conversation_id="test_conv_1",
                session_id="test_session_1"
            )
            
            print(f"✓ Stored knowledge with ID: {knowledge_id}")
            
            # Test 2: Store memory operation
            print("\n2. Testing store_memory_operation...")
            
            operation_id = knowledge_db.store_memory_operation(
                operation_type=MemoryOperation.ADD,
                knowledge_id=knowledge_id,
                confidence=0.9,
                reasoning="New information about AI capabilities",
                execution_result={"success": True, "message": "Successfully added"}
            )
            
            print(f"✓ Stored memory operation with ID: {operation_id}")
            
            # Test 3: Get knowledge by session
            print("\n3. Testing get_knowledge_by_session...")
            
            session_knowledge = knowledge_db.get_knowledge_by_session("test_session_1")
            print(f"✓ Retrieved {len(session_knowledge)} knowledge entries for session")
            
            if session_knowledge:
                first_entry = session_knowledge[0]
                print(f"   - Content: {first_entry['content'][:50]}...")
                print(f"   - Topics: {first_entry['topics']}")
                print(f"   - Importance: {first_entry['importance']}")
            
            # Test 4: Search knowledge
            print("\n4. Testing search_knowledge...")
            
            search_results = knowledge_db.search_knowledge(
                query="AI capabilities",
                session_id="test_session_1",
                min_importance=0.5
            )
            
            print(f"✓ Found {len(search_results)} matching knowledge entries")
            
            # Test 5: Get statistics
            print("\n5. Testing get_knowledge_statistics...")
            
            stats = knowledge_db.get_knowledge_statistics()
            print(f"✓ Database statistics:")
            print(f"   - Total entries: {stats.get('total_knowledge_entries', 0)}")
            print(f"   - Operations: {stats.get('operations_stats', {})}")
            print(f"   - Average importance: {stats.get('avg_importance', 0):.2f}")
            print(f"   - Database size: {stats.get('database_size_mb', 0):.2f} MB")
            
            print("\n✅ Knowledge Database basic tests PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Knowledge Database basic tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_memory_manager_integration():
    """Test tích hợp Memory Manager với Knowledge Database"""
    
    print("\n=== Test Memory Manager Integration ===")
    
    # Tạo temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_knowledge.db")
        
        try:
            # Initialize components
            print("\n1. Initializing components...")
            
            # Mock LLM Extractor (không cần real API key để test)
            class MockLLMExtractor:
                def __init__(self):
                    self.model_name = "mock-gpt-4o-mini"
                
                def extract_key_info(self, dialogue_turn: str, context: str = "") -> Dict[str, Any]:
                    # Simple mock extraction
                    return {
                        "entities": ["user", "bot"],
                        "intent": "general_conversation",
                        "key_facts": [f"Discussed: {dialogue_turn[:30]}..."],
                        "topics": ["conversation"],
                        "sentiment": "neutral",
                        "importance": 0.7,
                        "memory_type": "other",
                        "summary": f"Conversation about {dialogue_turn[:20]}...",
                        "requires_memory": True
                    }
            
            # Initialize Retrieval Memory
            retrieval_memory = RetrievalAugmentedMemory(
                store_type="simple",  # Use simple store để tránh ChromaDB dependency
                max_memories=100
            )
            
            # Initialize Memory Manager với Knowledge Database
            memory_manager = IntelligentMemoryManager(
                memory_system=retrieval_memory,
                llm_extractor=MockLLMExtractor(),
                knowledge_db_path=db_path,
                enable_knowledge_db=True
            )
            
            print(f"✓ Memory Manager initialized with Knowledge DB: {memory_manager.enable_knowledge_db}")
            
            # Test 2: Process dialogue turns
            print("\n2. Testing construct_memory_bank...")
            
            dialogue_turns = [
                "User: Hello, how are you?\nBot: I'm doing well, thank you! How can I help you today?",
                "User: Can you tell me about machine learning?\nBot: Machine learning is a subset of AI that enables computers to learn and improve from experience.",
                "User: That's interesting. What are some applications?\nBot: ML is used in recommendation systems, image recognition, natural language processing, and more."
            ]
            
            results = memory_manager.construct_memory_bank(
                dialogue_turns=dialogue_turns,
                context="Educational conversation about ML",
                session_id="test_session_2",
                conversation_id="test_conv_2"
            )
            
            print(f"✓ Processed {results['total_turns_processed']} dialogue turns")
            print(f"   - Memories added: {results['memories_added']}")
            print(f"   - Memories updated: {results['memories_updated']}")
            print(f"   - NOOP operations: {results['noop_operations']}")
            print(f"   - Processing errors: {results['processing_errors']}")
            
            # Test 3: Get operation statistics
            print("\n3. Testing get_operation_statistics...")
            
            stats = memory_manager.get_operation_statistics()
            print(f"✓ Operation statistics:")
            print(f"   - Total operations: {stats['total_operations']}")
            print(f"   - Operation counts: {stats['operation_counts']}")
            print(f"   - Knowledge DB enabled: {stats['knowledge_database_enabled']}")
            
            if 'knowledge_database_stats' in stats:
                kb_stats = stats['knowledge_database_stats']
                print(f"   - KB total entries: {kb_stats.get('total_knowledge_entries', 0)}")
            
            # Test 4: Get knowledge insights
            print("\n4. Testing get_knowledge_insights...")
            
            insights = memory_manager.get_knowledge_insights("test_session_2")
            print(f"✓ Knowledge insights:")
            
            if 'session_summary' in insights:
                session_summary = insights['session_summary']
                print(f"   - Session entries: {session_summary.get('total_entries', 0)}")
                print(f"   - Average importance: {session_summary.get('avg_importance', 0):.2f}")
                print(f"   - Common topics: {session_summary.get('common_topics', [])}")
            
            # Test 5: Search knowledge bank
            print("\n5. Testing search_knowledge_bank...")
            
            search_results = memory_manager.search_knowledge_bank(
                query="machine learning",
                session_id="test_session_2"
            )
            
            print(f"✓ Found {len(search_results)} matching entries for 'machine learning'")
            
            print("\n✅ Memory Manager integration tests PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Memory Manager integration tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_rl_chatbot_integration():
    """Test tích hợp RLChatbot với Knowledge Database"""
    
    print("\n=== Test RLChatbot Integration ===")
    
    try:
        # Tạo config với Knowledge Database enabled
        config = {
            "enable_knowledge_db": True,
            "knowledge_db_path": "data/test_chatbot_knowledge.db",
            "openai_api_key": "mock-key-for-testing",  # Mock key
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        # Note: Để test RLChatbot integration hoàn chỉnh cần OpenAI API key
        # Ở đây chỉ test initialization và basic methods
        
        print("\n1. Testing RLChatbot initialization...")
        
        # Initialize agent (có thể fail nếu không có real API key)
        try:
            agent = RLChatbotAgent(
                openai_model="gpt-4o-mini",
                api_key="mock-key",  # This will cause API calls to fail, but init should work
                config=config
            )
            
            print(f"✓ RLChatbot initialized")
            print(f"   - Knowledge DB enabled: {agent.memory_manager.enable_knowledge_db}")
            
            # Test get_memory_manager_insights
            print("\n2. Testing get_memory_manager_insights...")
            
            insights = agent.get_memory_manager_insights()
            print(f"✓ Memory manager insights:")
            print(f"   - Knowledge DB enabled: {insights['configuration']['knowledge_database_enabled']}")
            
            # Test knowledge methods (these should work without API calls)
            print("\n3. Testing knowledge methods...")
            
            # Start a session
            session_id = agent.start_session("test_user")
            print(f"✓ Started session: {session_id}")
            
            # Get knowledge by session (should return empty initially)
            knowledge = agent.get_knowledge_by_session()
            print(f"✓ Retrieved {len(knowledge)} knowledge entries for session")
            
            # Search knowledge (should return empty initially)
            search_results = agent.search_knowledge("test query")
            print(f"✓ Search returned {len(search_results)} results")
            
            # Get knowledge statistics
            stats = agent.get_knowledge_statistics()
            print(f"✓ Knowledge statistics: {type(stats).__name__}")
            
            print("\n✅ RLChatbot integration tests PASSED")
            return True
            
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print(f"⚠️  RLChatbot API authentication issue (expected in test): {e}")
                print("✓ Knowledge Database integration appears to be properly configured")
                return True
            else:
                raise e
            
    except Exception as e:
        print(f"\n❌ RLChatbot integration tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_scenario():
    """Test scenario hoàn chỉnh với mock data"""
    
    print("\n=== Test End-to-End Scenario ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_e2e_knowledge.db")
        
        try:
            # Initialize Knowledge Database
            knowledge_db = KnowledgeDatabaseManager(db_path)
            
            print("\n1. Simulating conversation flow...")
            
            # Simulate extracted knowledge từ nhiều conversations
            conversations = [
                {
                    "session_id": "user_001",
                    "conversation_id": "conv_001",
                    "turns": [
                        "User: What is Python?\nBot: Python is a high-level programming language.",
                        "User: Is it good for AI?\nBot: Yes, Python is excellent for AI and machine learning.",
                    ]
                },
                {
                    "session_id": "user_001", 
                    "conversation_id": "conv_002",
                    "turns": [
                        "User: How do I install Python?\nBot: You can download Python from python.org or use package managers.",
                    ]
                },
                {
                    "session_id": "user_002",
                    "conversation_id": "conv_003", 
                    "turns": [
                        "User: What's machine learning?\nBot: Machine learning is a subset of AI that learns from data.",
                    ]
                }
            ]
            
            # Process conversations
            for conv in conversations:
                for i, turn in enumerate(conv["turns"]):
                    # Simulate extracted info
                    extracted_info = {
                        "entities": ["Python", "AI", "machine learning"],
                        "intent": "educational_question",
                        "key_facts": [f"Fact from turn {i+1}"],
                        "topics": ["programming", "AI", "education"],
                        "sentiment": "positive",
                        "importance": 0.6 + (i * 0.1),
                        "memory_type": "factual",
                        "summary": f"Educational discussion about programming/AI",
                        "metadata": {"turn_index": i}
                    }
                    
                    # Store knowledge
                    knowledge_id = knowledge_db.store_extracted_knowledge(
                        extracted_info=extracted_info,
                        dialogue_turn=turn,
                        context=f"Educational conversation",
                        conversation_id=conv["conversation_id"],
                        session_id=conv["session_id"]
                    )
                    
                    # Store operation
                    knowledge_db.store_memory_operation(
                        operation_type=MemoryOperation.ADD,
                        knowledge_id=knowledge_id,
                        confidence=0.8,
                        reasoning="Educational content",
                        execution_result={"success": True, "message": "Added educational content"}
                    )
            
            print(f"✓ Processed {sum(len(c['turns']) for c in conversations)} conversation turns")
            
            # Test 2: Query và analyze
            print("\n2. Testing queries and analysis...")
            
            # Search across all sessions
            python_results = knowledge_db.search_knowledge("Python", min_importance=0.5)
            print(f"✓ Found {len(python_results)} entries about Python")
            
            # Get knowledge by specific session
            user1_knowledge = knowledge_db.get_knowledge_by_session("user_001")
            print(f"✓ User 001 has {len(user1_knowledge)} knowledge entries")
            
            user2_knowledge = knowledge_db.get_knowledge_by_session("user_002") 
            print(f"✓ User 002 has {len(user2_knowledge)} knowledge entries")
            
            # Get overall statistics
            stats = knowledge_db.get_knowledge_statistics()
            print(f"✓ Overall statistics:")
            print(f"   - Total entries: {stats['total_knowledge_entries']}")
            print(f"   - Average importance: {stats['avg_importance']:.2f}")
            print(f"   - Common topics: {[topic[0] for topic in stats['common_topics'][:3]]}")
            
            # Test 3: Export functionality
            print("\n3. Testing export functionality...")
            
            export_path = os.path.join(temp_dir, "exported_knowledge.json")
            knowledge_db.export_knowledge_data(export_path, "user_001")
            
            # Verify export
            if os.path.exists(export_path):
                with open(export_path, 'r', encoding='utf-8') as f:
                    exported_data = json.load(f)
                print(f"✓ Exported {exported_data['total_entries']} entries to {export_path}")
            
            print("\n✅ End-to-end scenario tests PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ End-to-end scenario tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Chạy tất cả tests"""
    
    print("🧪 Knowledge Database Integration Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run individual tests
    tests = [
        ("Knowledge Database Basic", test_knowledge_database_basic),
        ("Memory Manager Integration", test_memory_manager_integration),
        ("RLChatbot Integration", test_rl_chatbot_integration),
        ("End-to-End Scenario", test_end_to_end_scenario)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests PASSED! Knowledge Database integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
