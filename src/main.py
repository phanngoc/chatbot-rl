"""
Main application cho RL Chatbot
"""

import argparse
import json
import os
from typing import Dict, Any
from agents.rl_chatbot import RLChatbotAgent


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration từ file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "model_name": "microsoft/DialoGPT-medium",
            "device": "cpu",
            "experience_buffer_size": 10000,
            "memory_store_type": "chroma",
            "max_memories": 5000,
            "consolidation_threshold": 100,
            "consolidation_interval": 24,
            "ewc_lambda": 1000.0,
            "meta_memory_size": 1000,
            "decay_function": "exponential",
            "decay_params": {"decay_rate": 0.05, "min_weight": 0.01},
            "weight_update_interval": 6,
            "learning_rate": 1e-4,
            "temperature": 0.8
        }


def interactive_chat(agent: RLChatbotAgent):
    """Interactive chat mode"""
    print("🤖 RL Chatbot đã sẵn sàng! Nhập 'exit' để thoát, 'status' để xem trạng thái.")
    print("=" * 60)
    
    # Bắt đầu conversation
    conversation_id = agent.start_conversation()
    print(f"Conversation ID: {conversation_id}")
    
    while True:
        try:
            # Lấy input từ user
            user_input = input("\n👤 Bạn: ").strip()
            
            if user_input.lower() == 'exit':
                print("👋 Tạm biệt!")
                break
            elif user_input.lower() == 'status':
                status = agent.get_system_status()
                print("\n📊 System Status:")
                print(f"- Total interactions: {status['performance_metrics']['total_interactions']}")
                print(f"- Positive feedback: {status['performance_metrics']['positive_feedback']}")
                print(f"- Negative feedback: {status['performance_metrics']['negative_feedback']}")
                print(f"- Avg response time: {status['performance_metrics']['avg_response_time']:.2f}s")
                print(f"- Experience buffer: {len(agent.experience_buffer.buffer)}/{agent.experience_buffer.max_size}")
                continue
            elif user_input.lower() == 'conversation':
                summary = agent.get_conversation_summary()
                print("\n💬 Conversation Summary:")
                print(f"- Total exchanges: {summary['total_exchanges']}")
                print(f"- Memories used: {summary['total_memories_used']}")
                continue
            elif user_input.lower().startswith('feedback'):
                # Feedback format: "feedback <experience_id> <score>"
                parts = user_input.split()
                if len(parts) >= 3:
                    exp_id = parts[1]
                    try:
                        score = float(parts[2])
                        success = agent.provide_feedback(exp_id, score)
                        print(f"✅ Feedback {'đã được ghi nhận' if success else 'không thành công'}")
                    except ValueError:
                        print("❌ Score phải là số (ví dụ: feedback abc123 0.8)")
                else:
                    print("❌ Format: feedback <experience_id> <score>")
                continue
            
            if not user_input:
                continue
            
            # Process message
            print("🤖 Đang suy nghĩ...")
            result = agent.process_message(user_input)
            
            # Hiển thị response
            print(f"🤖 Bot: {result['response']}")
            
            # Hiển thị thông tin bổ sung
            print(f"📝 Experience ID: {result['experience_id']}")
            print(f"🧠 Memories used: {result['relevant_memories_count']}")
            print(f"⏱️  Response time: {result['response_time_ms']:.1f}ms")
            
            # Hỏi feedback
            feedback_input = input("\n💭 Đánh giá response (1-5, hoặc Enter để bỏ qua): ").strip()
            if feedback_input:
                try:
                    feedback_score = (float(feedback_input) - 3) / 2  # Convert 1-5 to -1 to 1
                    agent.provide_feedback(result['experience_id'], feedback_score)
                    print("✅ Cảm ơn feedback của bạn!")
                except ValueError:
                    print("❌ Feedback không hợp lệ")
        
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


def batch_training_mode(agent: RLChatbotAgent, training_data_path: str):
    """Batch training mode từ file data"""
    print(f"🎓 Bắt đầu batch training từ: {training_data_path}")
    
    if not os.path.exists(training_data_path):
        print(f"❌ File không tồn tại: {training_data_path}")
        return
    
    # Load training data
    with open(training_data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"📊 Loaded {len(training_data)} training samples")
    
    # Process each sample
    for i, sample in enumerate(training_data):
        user_message = sample.get("user_message", "")
        context = sample.get("context", "")
        expected_response = sample.get("expected_response", "")
        reward = sample.get("reward", 0.0)
        
        if not user_message:
            continue
        
        # Process message
        result = agent.process_message(user_message, context)
        
        # Provide feedback based on expected response (simplified)
        if expected_response:
            # Simple similarity check (trong thực tế sẽ phức tạp hơn)
            similarity = len(set(result['response'].lower().split()) & 
                           set(expected_response.lower().split())) / \
                        len(set(expected_response.lower().split()))
            feedback_score = (similarity - 0.5) * 2  # Convert to [-1, 1]
        else:
            feedback_score = reward
        
        agent.provide_feedback(result['experience_id'], feedback_score)
        
        if (i + 1) % 10 == 0:
            print(f"✅ Processed {i + 1}/{len(training_data)} samples")
    
    print("🎉 Batch training completed!")
    
    # Show final stats
    status = agent.get_system_status()
    print(f"📊 Final stats:")
    print(f"- Total interactions: {status['performance_metrics']['total_interactions']}")
    print(f"- Experience buffer size: {len(agent.experience_buffer.buffer)}")


def evaluation_mode(agent: RLChatbotAgent, test_data_path: str):
    """Evaluation mode"""
    print(f"🧪 Bắt đầu evaluation từ: {test_data_path}")
    
    if not os.path.exists(test_data_path):
        print(f"❌ File không tồn tại: {test_data_path}")
        return
    
    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"📊 Loaded {len(test_data)} test samples")
    
    results = []
    total_score = 0.0
    
    for i, sample in enumerate(test_data):
        user_message = sample.get("user_message", "")
        context = sample.get("context", "")
        expected_response = sample.get("expected_response", "")
        
        if not user_message:
            continue
        
        # Generate response
        result = agent.process_message(user_message, context)
        
        # Calculate score (simplified)
        if expected_response:
            response_words = set(result['response'].lower().split())
            expected_words = set(expected_response.lower().split())
            
            if expected_words:
                precision = len(response_words & expected_words) / len(response_words) if response_words else 0
                recall = len(response_words & expected_words) / len(expected_words)
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1_score = 0.0
        else:
            f1_score = 0.5  # Default neutral score
        
        total_score += f1_score
        
        results.append({
            "user_message": user_message,
            "generated_response": result['response'],
            "expected_response": expected_response,
            "f1_score": f1_score,
            "response_time_ms": result['response_time_ms'],
            "memories_used": result['relevant_memories_count']
        })
        
        if (i + 1) % 10 == 0:
            print(f"✅ Evaluated {i + 1}/{len(test_data)} samples")
    
    # Calculate final metrics
    avg_score = total_score / len(results) if results else 0.0
    avg_response_time = sum(r['response_time_ms'] for r in results) / len(results) if results else 0.0
    avg_memories_used = sum(r['memories_used'] for r in results) / len(results) if results else 0.0
    
    print("\n🎯 Evaluation Results:")
    print(f"- Average F1 Score: {avg_score:.3f}")
    print(f"- Average Response Time: {avg_response_time:.1f}ms")
    print(f"- Average Memories Used: {avg_memories_used:.1f}")
    
    # Save detailed results
    results_path = test_data_path.replace('.json', '_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "avg_f1_score": avg_score,
                "avg_response_time_ms": avg_response_time,
                "avg_memories_used": avg_memories_used,
                "total_samples": len(results)
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Detailed results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="RL Chatbot Application")
    parser.add_argument("--config", default="configs/default.json", help="Config file path")
    parser.add_argument("--mode", choices=["interactive", "train", "eval"], default="interactive", help="Running mode")
    parser.add_argument("--data", help="Data file for training/evaluation")
    parser.add_argument("--load", help="Load agent state from file")
    parser.add_argument("--save", help="Save agent state to file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"📋 Loaded config from: {args.config}")
    
    # Initialize agent
    print("🚀 Initializing RL Chatbot Agent...")
    agent = RLChatbotAgent(
        model_name=config.get("model_name", "microsoft/DialoGPT-medium"),
        device=config.get("device", "cpu"),
        config=config
    )
    
    # Load state if specified
    if args.load:
        print(f"📂 Loading agent state from: {args.load}")
        success = agent.load_agent_state(args.load)
        if success:
            print("✅ Agent state loaded successfully")
        else:
            print("❌ Failed to load agent state")
    
    # Run trong mode được chọn
    try:
        if args.mode == "interactive":
            interactive_chat(agent)
        elif args.mode == "train":
            if not args.data:
                print("❌ Training mode requires --data argument")
                return
            batch_training_mode(agent, args.data)
        elif args.mode == "eval":
            if not args.data:
                print("❌ Evaluation mode requires --data argument")
                return
            evaluation_mode(agent, args.data)
    
    except Exception as e:
        print(f"❌ Error during execution: {e}")
    
    finally:
        # Save state if specified
        if args.save:
            print(f"💾 Saving agent state to: {args.save}")
            success = agent.save_agent_state(args.save)
            if success:
                print("✅ Agent state saved successfully")
            else:
                print("❌ Failed to save agent state")


if __name__ == "__main__":
    main()
