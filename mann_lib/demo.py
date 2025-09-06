#!/usr/bin/env python3
"""
Demo script cho MANN CLI Chatbot
Thể hiện các tính năng chính của hệ thống
"""

import asyncio
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def demo_basic_conversation():
    """Demo cuộc trò chuyện cơ bản"""
    print("🎭 Demo: Basic Conversation")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Simulate conversation
        conversations = [
            "Xin chào, tôi tên là Ngọc",
            "Tôi 25 tuổi và đang làm việc tại Hà Nội",
            "Tôi thích lập trình Python và học về AI",
            "Bạn có nhớ tên tôi không?",
            "Tôi đang học về Machine Learning",
            "Bạn biết gì về sở thích của tôi?",
            "Tôi muốn tạo một chatbot thông minh",
            "Hôm nay trời đẹp, tôi đi dạo công viên"
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 User {i}: {user_input}")
            
            start_time = time.time()
            response = await chatbot.process_user_input(user_input)
            processing_time = time.time() - start_time
            
            print(f"🤖 Bot: {response}")
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            
            # Small delay for demo effect
            await asyncio.sleep(1)
        
        # Show statistics
        stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Session Statistics:")
        print(f"  Total queries: {stats.get('total_queries', 0)}")
        print(f"  Memories created: {stats.get('total_memories_created', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        print(f"  Total retrievals: {stats.get('total_retrievals', 0)}")
        print(f"  Total writes: {stats.get('total_writes', 0)}")
        print(f"  Memory matrix norm: {stats.get('memory_matrix_norm', 0):.4f}")
        print(f"  Ŵ norm: {stats.get('W_hat_norm', 0):.4f}")
        print(f"  V̂ norm: {stats.get('V_hat_norm', 0):.4f}")
        
    finally:
        await chatbot.shutdown()


async def demo_memory_search():
    """Demo tìm kiếm memory"""
    print("\n🔍 Demo: Memory Search")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Add some test memories first
        test_memories = [
            "Tôi tên là Ngọc, 25 tuổi",
            "Tôi thích lập trình Python và JavaScript",
            "Tôi đang học về AI và Machine Learning",
            "Tôi sống ở Hà Nội và làm việc tại công ty ABC",
            "Tôi thích đọc sách và chơi game",
            "Tôi muốn trở thành một AI Engineer",
            "Hôm nay tôi đi mua sắm ở trung tâm thương mại",
            "Tôi có một con mèo tên là Mimi"
        ]
        
        print("📝 Adding test memories...")
        for memory in test_memories:
            await chatbot.process_user_input(memory)
            await asyncio.sleep(0.5)
        
        # Search for different topics
        search_queries = [
            "tên",
            "lập trình",
            "AI",
            "Hà Nội",
            "mèo",
            "sở thích"
        ]
        
        print("\n🔍 Searching memories...")
        for query in search_queries:
            print(f"\nQuery: '{query}'")
            results = await chatbot.search_memories(query, top_k=3)
            
            if results:
                print(f"  Found {len(results)} memories:")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['content'][:60]}...")
                    print(f"       Similarity: {result.get('similarity', 0):.3f}")
                    print(f"       Importance: {result.get('importance_weight', 0):.2f}")
            else:
                print("  No memories found.")
    
    finally:
        await chatbot.shutdown()


async def demo_memory_management():
    """Demo quản lý memory"""
    print("\n💾 Demo: Memory Management")
    print("=" * 50)
    
    config = MANNConfig()
    config.memory_size = 10  # Small size for demo
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Add memories until capacity is reached
        print("📝 Adding memories to test capacity management...")
        
        for i in range(15):  # More than memory_size
            memory_content = f"Memory {i+1}: This is test memory number {i+1}"
            await chatbot.process_user_input(memory_content)
            await asyncio.sleep(0.1)
        
        # Show memory statistics
        stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Memory Statistics:")
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0):.2%}")
        print(f"  Max capacity: {config.memory_size}")
        
        # Show memory contents
        print(f"\n📋 Current Memories:")
        for i, memory in enumerate(chatbot.mann_model.memory_bank, 1):
            print(f"  {i}. {memory.content[:50]}...")
            print(f"     Importance: {memory.importance_weight:.2f}")
            print(f"     Usage count: {memory.usage_count}")
    
    finally:
        await chatbot.shutdown()


async def demo_health_monitoring():
    """Demo health monitoring"""
    print("\n🏥 Demo: Health Monitoring")
    print("=" * 50)
    
    config = MANNConfig()
    config.enable_monitoring = True
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        # Perform some operations
        print("🔄 Performing operations to generate metrics...")
        
        for i in range(5):
            await chatbot.process_user_input(f"Test operation {i+1}")
            await asyncio.sleep(0.5)
        
        # Check health
        print("\n🏥 Health Check:")
        health = await chatbot.health_check()
        print(f"  Status: {health.get('status', 'unknown')}")
        
        checks = health.get('checks', {})
        for check_name, check_data in checks.items():
            status = check_data.get('status', 'unknown')
            status_emoji = "✅" if status == "healthy" else "❌"
            print(f"  {status_emoji} {check_name}: {status}")
        
        # Show performance stats
        if chatbot.monitor:
            perf_stats = chatbot.monitor.get_performance_stats()
            print(f"\n📈 Performance Statistics:")
            print(f"  Total queries: {perf_stats.get('total_queries', 0)}")
            print(f"  Average processing time: {perf_stats.get('avg_processing_time', 0):.3f}s")
            print(f"  Error rate: {perf_stats.get('error_rate', 0):.2%}")
            print(f"  Memory utilization: {perf_stats.get('memory_utilization', 0):.2%}")
    
    finally:
        await chatbot.shutdown()


async def demo_external_working_memory():
    """Demo External Working Memory features"""
    print("\n🧠 Demo: External Working Memory")
    print("=" * 50)
    
    config = MANNConfig()
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        print("🔬 Testing External Working Memory Operations...")
        
        # Test sequences to demonstrate learning
        test_sequences = [
            "Tôi tên là Ngọc và thích lập trình",
            "Tôi đang học về AI và Machine Learning", 
            "Tôi sống ở Hà Nội và làm việc tại ABC",
            "Tôi có sở thích đọc sách và chơi game",
            "Tôi muốn trở thành AI Engineer"
        ]
        
        print("\n📝 Processing test sequences...")
        for i, sequence in enumerate(test_sequences, 1):
            print(f"\n👤 Input {i}: {sequence}")
            
            start_time = time.time()
            response = await chatbot.process_user_input(sequence)
            processing_time = time.time() - start_time
            
            print(f"🤖 Response: {response}")
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            
            # Show memory statistics after each input
            stats = await chatbot.get_memory_statistics()
            print(f"📊 Memory stats: retrievals={stats.get('total_retrievals', 0)}, writes={stats.get('total_writes', 0)}")
            
            await asyncio.sleep(0.5)
        
        # Test memory search
        print("\n🔍 Testing Memory Search...")
        search_queries = ["tên", "lập trình", "AI", "Hà Nội", "sở thích"]
        
        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = await chatbot.search_memories(query, top_k=2)
            
            if results:
                print(f"  Found {len(results)} relevant memories:")
                for j, result in enumerate(results, 1):
                    print(f"    {j}. {result['content'][:60]}...")
                    print(f"       Similarity: {result.get('similarity', 0):.3f}")
            else:
                print("  No relevant memories found.")
        
        # Show final statistics
        final_stats = await chatbot.get_memory_statistics()
        print(f"\n📊 Final External Working Memory Statistics:")
        print(f"  Total queries: {final_stats.get('total_queries', 0)}")
        print(f"  Total retrievals: {final_stats.get('total_retrievals', 0)}")
        print(f"  Total writes: {final_stats.get('total_writes', 0)}")
        print(f"  Memory matrix norm: {final_stats.get('memory_matrix_norm', 0):.4f}")
        print(f"  Ŵ matrix norm: {final_stats.get('W_hat_norm', 0):.4f}")
        print(f"  V̂ matrix norm: {final_stats.get('V_hat_norm', 0):.4f}")
        print(f"  Memory utilization: {final_stats.get('memory_utilization', 0):.2%}")
        
        print(f"\n✅ External Working Memory Demo Complete!")
        print(f"   - Memory Write: μ̇ᵢ = -zᵢμᵢ + cwzᵢa + zᵢŴqμᵀ")
        print(f"   - Memory Read: Mr = μz, z = softmax(μᵀq)")
        print(f"   - NN Output: uad = -Ŵᵀ(σ(V̂ᵀx̃ + b̂v) + Mr) - b̂w")
        
    finally:
        await chatbot.shutdown()


async def demo_ppo_training():
    """Demo PPO training with memory-augmented network"""
    print("\n🎯 Demo: PPO Training with Memory")
    print("=" * 50)
    
    import torch
    import numpy as np
    import os
    from standalone_mann.mann_core import MemoryAugmentedNetwork
    
    # Clear any existing debug log
    debug_log_path = "debug_reward_process.log"
    if os.path.exists(debug_log_path):
        os.remove(debug_log_path)
        print(f"🗑️  Cleared existing debug log: {debug_log_path}")
    
    # Create a simple MANN model for testing
    print("🧠 Initializing MANN model for PPO training...")
    
    # Model configuration
    input_size = 64
    hidden_size = 128
    memory_size = 20  # Reduced for cleaner debugging
    memory_dim = 64
    output_size = 1000  # Vocabulary size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    try:
        # Initialize model
        mann_model = MemoryAugmentedNetwork(
            input_size=input_size,
            hidden_size=hidden_size, 
            memory_size=memory_size,
            memory_dim=memory_dim,
            output_size=output_size,
            device=device
        ).to(device)
        
        print(f"  Model initialized with {sum(p.numel() for p in mann_model.parameters())} parameters")
        
        # Clear any existing memory bank
        mann_model.memory_bank = []
        print("  Cleared existing memory bank")
        
        # Add comprehensive test memories for better debugging
        print("\n📝 Adding comprehensive test memories...")
        test_memories = [
            # Programming concepts
            ("Python là ngôn ngữ lập trình cao cấp, dễ học và mạnh mẽ", "programming_context", ["python", "programming", "language"]),
            ("JavaScript là ngôn ngữ web phổ biến cho frontend và backend", "web_context", ["javascript", "web", "frontend"]),
            ("Java là ngôn ngữ hướng đối tượng mạnh mẽ cho enterprise", "enterprise_context", ["java", "oop", "enterprise"]),
            
            # AI/ML concepts
            ("Machine Learning là nhánh của AI sử dụng dữ liệu để học", "ml_context", ["ml", "ai", "data"]),
            ("Deep Learning sử dụng neural network nhiều lớp", "dl_context", ["deep_learning", "neural_network"]),
            ("Natural Language Processing xử lý ngôn ngữ tự nhiên", "nlp_context", ["nlp", "language", "processing"]),
            
            # Frameworks
            ("PyTorch là framework deep learning linh hoạt của Facebook", "framework_context", ["pytorch", "deep_learning", "facebook"]),
            ("TensorFlow là platform ML mở của Google", "tf_context", ["tensorflow", "ml", "google"]),
            ("Scikit-learn là thư viện ML cơ bản cho Python", "sklearn_context", ["sklearn", "ml", "python"]),
            
            # Algorithms
            ("Reinforcement Learning học thông qua reward và punishment", "rl_context", ["rl", "reward", "learning"]),
            ("PPO là thuật toán policy optimization ổn định", "ppo_context", ["ppo", "optimization", "policy"]),
            ("Q-Learning học value function thông qua exploration", "qlearning_context", ["qlearning", "value", "exploration"])
        ]
        
        for i, (content, context, tags) in enumerate(test_memories):
            memory_id = mann_model.add_memory(content, context, tags, importance_weight=1.0 + i*0.1)
            print(f"  [{i+1:2d}] Added: {memory_id[:8]}... - {content[:40]}...")
        
        # Generate comprehensive training data
        print("\n🎲 Generating comprehensive training data...")
        questions = [
            "Python có ưu điểm gì?",
            "Machine Learning hoạt động như thế nao?", 
            "PyTorch khác TensorFlow như thế nào?",
            "Reinforcement Learning là gì?",
            "PPO algorithm có gì đặc biệt?",
            "Deep Learning và Machine Learning khác nhau ra sao?",
            "JavaScript dùng để làm gì?",
            "Natural Language Processing giải quyết vấn đề gì?"
        ]
        
        reference_answers = [
            "Python là ngôn ngữ lập trình cao cấp, dễ học và mạnh mẽ",
            "Machine Learning là nhánh của AI sử dụng dữ liệu để học",
            "PyTorch là framework deep learning linh hoạt của Facebook", 
            "Reinforcement Learning học thông qua reward và punishment",
            "PPO là thuật toán policy optimization ổn định",
            "Deep Learning sử dụng neural network nhiều lớp",
            "JavaScript là ngôn ngữ web phổ biến cho frontend và backend",
            "Natural Language Processing xử lý ngôn ngữ tự nhiên"
        ]
        
        # Create input tensors
        batch_size = len(questions)
        seq_len = 10
        input_tensors = torch.randn(batch_size, seq_len, input_size, device=device)
        
        print(f"  Created {batch_size} training samples")
        
        # Test PPO forward pass
        print("\n🔄 Testing PPO forward pass...")
        forward_results = mann_model.ppo_forward_with_memory(
            input_tensors, questions, generate_answers=True
        )
        
        print(f"  Forward pass completed:")
        print(f"    Logits shape: {forward_results['logits'].shape}")
        print(f"    Values shape: {forward_results['values'].shape}")
        print(f"    Memory context shape: {forward_results['memory_context'].shape}")
        print(f"    Retrieved memories: {len(forward_results['retrieved_memories'])}")
        
        # Generate answers using current policy (concise output)
        print("\n🎯 Generating answers with current policy...")
        generated_answers = []
        
        for i, question in enumerate(questions):
            answer, memory_info = mann_model.generate_answer_with_ppo(
                question, input_tensors[i], max_length=30  # Shorter for cleaner logs
            )
            generated_answers.append(answer)
            print(f"  [{i+1}] Q: {question[:35]}...")
            print(f"      A: {answer[:50]}...")
            print(f"      Memories: {len(memory_info)}")
        
        # Test reward computation
        print("\n🏆 Computing answer rewards...")
        rewards = mann_model.compute_answer_rewards(
            generated_answers, reference_answers, questions
        )
        reward_stats = {
            'values': rewards.numpy().tolist(),
            'mean': rewards.mean().item(),
            'std': rewards.std().item(),
            'min': rewards.min().item(),
            'max': rewards.max().item()
        }
        print(f"  Rewards: {[f'{r:.3f}' for r in reward_stats['values']]}")
        print(f"  Stats: mean={reward_stats['mean']:.3f}, std={reward_stats['std']:.3f}")
        
        # Perform PPO training step with detailed debugging
        print(f"\n🚀 Performing PPO training step (epochs=2, lr=3e-4)...")
        print(f"   📊 Check '{debug_log_path}' for detailed reward process debugging")
        
        training_stats = mann_model.ppo_update(
            questions=questions,
            generated_answers=generated_answers,
            reference_answers=reference_answers,
            input_tensors=input_tensors,
            learning_rate=3e-4,
            epochs=2  # Reduced for demo
        )
        
        print(f"  ✅ Training completed:")
        print(f"    📉 Loss: {training_stats['avg_loss']:.4f}")
        print(f"    🏆 Reward: {training_stats['avg_reward']:.4f}")
        print(f"    📈 Advantage: {training_stats['avg_advantage']:.4f}")
        print(f"    🔀 Entropy: {training_stats['policy_entropy']:.4f}")
        print(f"    ⚖️  Importance ratio: {training_stats['importance_ratio']:.4f}")
        print(f"    💾 Memories retrieved: {training_stats['memories_retrieved']}")
        
        # Test after training (concise output)
        print("\n🔄 Testing policy after training...")
        new_generated_answers = []
        
        print(f"  Post-training answers (first 3 samples):")
        for i, question in enumerate(questions[:3]):  # Show only first 3 for brevity
            answer, memory_info = mann_model.generate_answer_with_ppo(
                question, input_tensors[i], max_length=30
            )
            new_generated_answers.append(answer)
            print(f"    [{i+1}] Q: {question[:30]}...")
            print(f"        A: {answer[:40]}...")
        
        # Generate all new answers for comparison
        for i, question in enumerate(questions[3:], 3):  # Complete remaining answers
            answer, memory_info = mann_model.generate_answer_with_ppo(
                question, input_tensors[i], max_length=30
            )
            new_generated_answers.append(answer)
        
        # Compare rewards
        new_rewards = mann_model.compute_answer_rewards(
            new_generated_answers, reference_answers, questions
        )
        
        print(f"\n📊 Training Results Summary:")
        print(f"  🎯 Pre-training  reward: {rewards.mean().item():.3f} (std: {rewards.std().item():.3f})")
        print(f"  🎯 Post-training reward: {new_rewards.mean().item():.3f} (std: {new_rewards.std().item():.3f})")
        improvement = (new_rewards.mean() - rewards.mean()).item()
        improvement_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        print(f"  {improvement_emoji} Improvement: {improvement:+.3f}")
        
        # Show compact memory statistics
        stats = mann_model.get_memory_statistics()
        print(f"\n💾 Final Memory Stats:")
        print(f"  📝 Memories: {stats['total_memories']} | Utilization: {stats['memory_utilization']:.1%}")
        print(f"  🔍 Retrievals: {stats['total_retrievals']} | Writes: {stats['total_writes']}")
        print(f"  🧠 Matrix norm: {stats['memory_matrix_norm']:.4f}")
        
        print(f"\n✅ PPO Training Demo Complete!")
        print(f"   📄 Debug details saved to: {debug_log_path}")
        print(f"   📊 Review the debug log to analyze reward computation process")
        
    except Exception as e:
        print(f"❌ PPO training demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_ppo_importance_ratio():
    """Demo PPO importance ratio calculation"""
    print("\n⚖️ Demo: PPO Importance Ratio Calculation")
    print("=" * 50)
    
    import torch
    from standalone_mann.mann_core import MemoryAugmentedNetwork
    
    print("🧮 Testing PPO importance ratio computation...")
    
    # Small model for testing
    mann_model = MemoryAugmentedNetwork(
        input_size=32,
        hidden_size=64, 
        memory_size=10,
        memory_dim=32,
        output_size=100
    )
    
    try:
        # Create test data
        hidden_state = torch.randn(64)  # hidden_size
        memory_context = torch.randn(32)  # memory_dim
        actions = torch.randint(0, 100, (5,))  # 5 action tokens
        
        print(f"  Hidden state shape: {hidden_state.shape}")
        print(f"  Memory context shape: {memory_context.shape}")
        print(f"  Actions: {actions.numpy()}")
        
        # Test importance ratio calculation
        importance_ratio = mann_model.memory_interface.compute_ppo_importance_ratio(
            hidden_state, memory_context, actions
        )
        
        print(f"  Importance ratios: {importance_ratio.detach().numpy()}")
        print(f"  Average importance ratio: {importance_ratio.mean().item():.3f}")
        print(f"  Min ratio: {importance_ratio.min().item():.3f}")
        print(f"  Max ratio: {importance_ratio.max().item():.3f}")
        
        # Test advantage computation
        rewards = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.85])
        values = torch.tensor([0.5, 0.6, 0.4, 0.3, 0.55])
        
        advantages, returns = mann_model.memory_interface.compute_advantages(rewards, values)
        
        print(f"\n📈 Advantage Computation:")
        print(f"  Rewards: {rewards.numpy()}")
        print(f"  Values: {values.numpy()}")
        print(f"  Advantages: {advantages.numpy()}")
        print(f"  Returns: {returns.numpy()}")
        
        # Test PPO loss computation
        loss_dict = mann_model.memory_interface.compute_ppo_loss(
            hidden_state, memory_context, actions, advantages, returns
        )
        
        print(f"\n📉 PPO Loss Components:")
        print(f"  Policy loss: {loss_dict['policy_loss'].item():.4f}")
        print(f"  Value loss: {loss_dict['value_loss'].item():.4f}")
        print(f"  Entropy loss: {loss_dict['entropy_loss'].item():.4f}")
        print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"  Entropy: {loss_dict['entropy'].item():.4f}")
        
        print(f"\n✅ PPO Importance Ratio Demo Complete!")
        
    except Exception as e:
        print(f"❌ PPO importance ratio demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_api_integration():
    """Demo API integration"""
    print("\n🌐 Demo: API Integration")
    print("=" * 50)
    
    from standalone_mann.mann_api import MANNClient
    
    # Note: This demo assumes API server is running
    print("📡 Testing API client (requires running API server)...")
    
    try:
        async with MANNClient("http://localhost:8000") as client:
            # Health check
            print("🏥 API Health Check:")
            health = await client.health_check()
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Memory count: {health.get('memory_count', 0)}")
            
            # Add memory via API
            print("\n📝 Adding memory via API:")
            memory_id = await client.add_memory(
                content="API test memory",
                context="api_demo",
                tags=["test", "api"],
                importance_weight=1.5
            )
            print(f"  Added memory: {memory_id}")
            
            # Search via API
            print("\n🔍 Searching via API:")
            results = await client.search_memories("test", top_k=3)
            print(f"  Found {len(results)} memories")
            
            # Process query via API
            print("\n🚀 Processing query via API:")
            response = await client.process_query("Hello from API", retrieve_memories=True)
            print(f"  Response: {response.get('output', 'No output')}")
            print(f"  Processing time: {response.get('processing_time', 0):.3f}s")
            
    except Exception as e:
        print(f"❌ API demo failed: {e}")
        print("💡 Make sure API server is running: python run_api.py")


async def main():
    """Run all demos"""
    print("🎪 MANN CLI Chatbot Demo Suite")
    print("=" * 60)
    
    demos = [
        ("Basic Conversation", demo_basic_conversation),
        ("External Working Memory", demo_external_working_memory),
        ("PPO Training", demo_ppo_training),
        ("PPO Importance Ratio", demo_ppo_importance_ratio),
        ("Memory Search", demo_memory_search),
        ("Memory Management", demo_memory_management),
        ("Health Monitoring", demo_health_monitoring),
        ("API Integration", demo_api_integration)
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🎬 Starting: {demo_name}")
            await demo_func()
            print(f"✅ Completed: {demo_name}")
        except Exception as e:
            print(f"❌ Failed: {demo_name} - {e}")
        
        print("\n" + "="*60)
        await asyncio.sleep(2)  # Pause between demos
    
    print("\n🎉 All demos completed!")
    print("\n💡 To run individual demos:")
    print("  python demo.py --demo basic")
    print("  python demo.py --demo external") 
    print("  python demo.py --demo ppo")
    print("  python demo.py --demo ppo-ratio")
    print("  python demo.py --demo search")
    print("  python demo.py --demo memory")
    print("  python demo.py --demo health")
    print("  python demo.py --demo api")
    print("\n🧪 Or run organized test suite:")
    print("  cd tests && python run_all_tests.py")
    print("  cd tests && python test_ppo_training.py  # PPO with CSV data")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MANN CLI Chatbot Demo")
    parser.add_argument("--demo", choices=["basic", "external", "ppo", "ppo-ratio", "search", "memory", "health", "api"], 
                       help="Run specific demo")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_map = {
            "basic": demo_basic_conversation,
            "external": demo_external_working_memory,
            "ppo": demo_ppo_training,
            "ppo-ratio": demo_ppo_importance_ratio,
            "search": demo_memory_search,
            "memory": demo_memory_management,
            "health": demo_health_monitoring,
            "api": demo_api_integration
        }
        
        print(f"🎬 Running demo: {args.demo}")
        asyncio.run(demo_map[args.demo]())
    else:
        asyncio.run(main())
