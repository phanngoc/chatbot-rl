#!/usr/bin/env python3
"""
Simple OpenAI API test cho RL Chatbot
"""

import os
import json
import sys
sys.path.append('../src')

from src.agents.rl_chatbot import RLChatbotAgent

def test_openai_connection():
    """Test basic OpenAI connection"""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is required!")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello! This is a connection test."}],
            max_tokens=30
        )
        
        print("✅ OpenAI API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def test_rl_chatbot():
    """Test RL Chatbot với OpenAI"""
    
    print("\n🤖 Testing RL Chatbot with OpenAI...")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../configs/default.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Initialize agent
    agent = RLChatbotAgent(
        openai_model=config.get("openai_model", "gpt-3.5-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        config=config
    )
    
    # Start conversation
    conversation_id = agent.start_conversation()
    print(f"📋 Started conversation: {conversation_id}")
    
    # Test messages
    test_messages = [
        "Xin chào! Bạn có thể giới thiệu về bản thân không?",
        "Bạn có thể nhớ được những gì chúng ta đã nói không?",
        "Hãy kể cho tôi một câu chuyện ngắn."
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test {i} ---")
        print(f"👤 User: {message}")
        
        try:
            result = agent.process_message(message)
            
            print(f"🤖 Bot: {result['response']}")
            print(f"📊 Memories used: {result['relevant_memories_count']}")
            print(f"⏱️  Response time: {result['response_time_ms']:.2f}ms")
            
            if result.get('openai_usage'):
                usage = result['openai_usage']
                print(f"💰 Tokens: {usage.get('total_tokens', 0)} (prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)})")
            
            if result.get('api_error'):
                print(f"⚠️  API Error: {result['api_error']}")
                
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    # Get system status
    print(f"\n📈 System Status:")
    status = agent.get_system_status()
    print(f"   - Model: {status['model_info']['openai_model']}")
    print(f"   - Total interactions: {status['performance_metrics']['total_interactions']}")
    print(f"   - Neural parameters: {status['model_info']['neural_parameters']:,}")

def main():
    """Main test function"""
    print("🧪 OpenAI RL Chatbot Test")
    print("=" * 30)
    
    # Test OpenAI connection first
    if not test_openai_connection():
        return
    
    # Test RL Chatbot
    try:
        test_rl_chatbot()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

