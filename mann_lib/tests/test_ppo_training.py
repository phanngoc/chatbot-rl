#!/usr/bin/env python3
"""
Test: PPO Training with Memory-Augmented Network
Comprehensive PPO training test with CSV data loading and detailed debugging
"""

import asyncio
import sys
import os
import csv
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from standalone_mann.mann_core import MemoryAugmentedNetwork


def load_training_data(csv_path):
    """Load training data from CSV file"""
    questions = []
    reference_answers = []
    categories = []
    difficulties = []
    importance_weights = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['question'].strip('"'))
            reference_answers.append(row['reference_answer'].strip('"'))
            categories.append(row['category'])
            difficulties.append(int(row['difficulty']))
            importance_weights.append(float(row['importance_weight']))
    
    return questions, reference_answers, categories, difficulties, importance_weights


def load_memory_bank_data(csv_path):
    """Load memory bank initialization data from CSV"""
    memories = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            memories.append({
                'content': row['content'].strip('"'),
                'context': row['context'],
                'tags': row['tags'].split(','),
                'importance_weight': float(row['importance_weight'])
            })
    
    return memories


def setup_debug_logging():
    """Setup debug logging and clear existing files"""
    debug_log_path = "debug_reward_process.log"
    if os.path.exists(debug_log_path):
        os.remove(debug_log_path)
        print(f"🗑️  Cleared existing debug log: {debug_log_path}")
    return debug_log_path


def print_training_flow_info():
    """Print information about the training flow"""
    print("🎯 PPO Training Flow:")
    print("   1️⃣  Initialize MANN model with external working memory")
    print("   2️⃣  Load memory bank data from CSV")
    print("   3️⃣  Load training questions and reference answers")
    print("   4️⃣  Forward pass: generate answers with current policy")
    print("   5️⃣  Compute rewards by comparing generated vs reference answers")
    print("   6️⃣  Compute advantages and returns for PPO")
    print("   7️⃣  Update policy using PPO loss (clipped importance ratio)")
    print("   8️⃣  Test improved policy and measure performance gain")
    print()


async def test_ppo_training():
    """Main PPO training test"""
    print("🎯 Test: PPO Training with Memory-Augmented Network")
    print("=" * 60)
    
    print_training_flow_info()
    debug_log_path = setup_debug_logging()
    
    # Model configuration
    input_size = 64
    hidden_size = 128
    memory_size = 25  # Adequate size for comprehensive testing
    memory_dim = 64
    output_size = 1000  # Vocabulary size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    try:
        # Step 1: Initialize MANN model
        print("\n1️⃣  Initializing MANN model...")
        mann_model = MemoryAugmentedNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            memory_dim=memory_dim,
            output_size=output_size,
            device=device
        ).to(device)
        
        param_count = sum(p.numel() for p in mann_model.parameters())
        print(f"   📊 Model initialized with {param_count:,} parameters")
        print(f"   🧠 Memory size: {memory_size}, Memory dim: {memory_dim}")
        
        # Clear existing memory bank
        mann_model.memory_bank = []
        
        # Step 2: Load memory bank data
        print("\n2️⃣  Loading memory bank from CSV...")
        data_dir = Path(__file__).parent / "data"
        memory_data = load_memory_bank_data(data_dir / "memory_bank_data.csv")
        
        print(f"   📚 Loading {len(memory_data)} memory entries...")
        for i, memory in enumerate(memory_data):
            memory_id = mann_model.add_memory(
                content=memory['content'],
                context=memory['context'],
                tags=memory['tags'],
                importance_weight=memory['importance_weight']
            )
            if i < 5:  # Show first 5 for brevity
                print(f"   [{i+1:2d}] {memory_id[:8]}... - {memory['content'][:45]}...")
        
        if len(memory_data) > 5:
            print(f"   ... and {len(memory_data) - 5} more memories")
        
        # Step 3: Load training data
        print("\n3️⃣  Loading training data from CSV...")
        questions, reference_answers, categories, difficulties, importance_weights = load_training_data(
            data_dir / "ppo_training_data.csv"
        )
        
        print(f"   📝 Loaded {len(questions)} training samples")
        print(f"   📊 Categories: {set(categories)}")
        print(f"   📈 Difficulty range: {min(difficulties)}-{max(difficulties)}")
        print(f"   ⭐ Importance range: {min(importance_weights):.1f}-{max(importance_weights):.1f}")
        
        # Sample questions for display
        print(f"   📋 Sample questions:")
        for i in range(min(3, len(questions))):
            print(f"      [{i+1}] {questions[i]}")
        
        # Create input tensors
        batch_size = len(questions)
        seq_len = 12
        input_tensors = torch.randn(batch_size, seq_len, input_size, device=device)
        
        # Step 4: Forward pass
        print(f"\n4️⃣  Forward pass with memory retrieval...")
        forward_results = mann_model.ppo_forward_with_memory(
            input_tensors, questions, generate_answers=True
        )
        
        print(f"   ✅ Forward pass completed:")
        print(f"      📊 Logits shape: {forward_results['logits'].shape}")
        print(f"      💰 Values shape: {forward_results['values'].shape}")
        print(f"      🧠 Memory context shape: {forward_results['memory_context'].shape}")
        print(f"      🔍 Retrieved memories: {len(forward_results['retrieved_memories'])}")
        
        # Step 5: Generate answers with current policy
        print(f"\n5️⃣  Generating answers with current policy...")
        generated_answers = []
        
        sample_size = min(8, len(questions))  # Show first 8 samples
        print(f"   🎯 Generating {len(questions)} answers (showing first {sample_size}):")
        
        for i, question in enumerate(questions):
            answer, memory_info = mann_model.generate_answer_with_ppo(
                question, input_tensors[i], max_length=25
            )
            generated_answers.append(answer)
            
            if i < sample_size:
                print(f"   [{i+1}] Q: {question[:40]}...")
                print(f"       A: {answer[:55]}...")
                print(f"       📚 Memories used: {len(memory_info)}")
        
        # Step 6: Compute rewards
        print(f"\n6️⃣  Computing answer rewards...")
        rewards = mann_model.compute_answer_rewards(
            generated_answers, reference_answers, questions
        )
        
        reward_stats = {
            'mean': rewards.mean().item(),
            'std': rewards.std().item(),
            'min': rewards.min().item(),
            'max': rewards.max().item(),
            'values': rewards.numpy().tolist()
        }
        
        print(f"   📊 Reward statistics:")
        print(f"      🎯 Mean: {reward_stats['mean']:.3f}")
        print(f"      📈 Std: {reward_stats['std']:.3f}")
        print(f"      📉 Range: [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]")
        print(f"   🔍 Sample rewards: {[f'{r:.3f}' for r in reward_stats['values'][:8]]}")
        
        # Step 7: PPO training
        print(f"\n7️⃣  PPO Training (detailed logging to {debug_log_path})...")
        print(f"   ⚙️  Hyperparameters: lr=3e-4, epochs=3, clip_ratio=0.2")
        
        training_stats = mann_model.ppo_update(
            questions=questions,
            generated_answers=generated_answers,
            reference_answers=reference_answers,
            input_tensors=input_tensors,
            learning_rate=3e-4,
            epochs=3  # More epochs for better training
        )
        
        print(f"   ✅ Training completed successfully!")
        print(f"      📉 Average loss: {training_stats['avg_loss']:.4f}")
        print(f"      🏆 Average reward: {training_stats['avg_reward']:.4f}")
        print(f"      📈 Average advantage: {training_stats['avg_advantage']:.4f}")
        print(f"      🎲 Policy entropy: {training_stats['policy_entropy']:.4f}")
        print(f"      ⚖️  Importance ratio: {training_stats['importance_ratio']:.4f}")
        print(f"      💾 Memories retrieved: {training_stats['memories_retrieved']}")
        
        # Step 8: Post-training evaluation
        print(f"\n8️⃣  Post-training evaluation...")
        new_generated_answers = []
        
        print(f"   🔄 Generating answers with updated policy...")
        for i, question in enumerate(questions):
            answer, memory_info = mann_model.generate_answer_with_ppo(
                question, input_tensors[i], max_length=25
            )
            new_generated_answers.append(answer)
        
        # Show sample improved answers
        print(f"   📋 Sample improved answers (first 4):")
        for i in range(min(4, len(questions))):
            print(f"   [{i+1}] Q: {questions[i][:35]}...")
            print(f"       Before: {generated_answers[i][:45]}...")
            print(f"       After:  {new_generated_answers[i][:45]}...")
        
        # Final reward comparison
        new_rewards = mann_model.compute_answer_rewards(
            new_generated_answers, reference_answers, questions
        )
        
        improvement = (new_rewards.mean() - rewards.mean()).item()
        improvement_pct = (improvement / rewards.mean().item()) * 100
        improvement_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        
        print(f"\n📊 Final Training Results:")
        print(f"   🎯 Pre-training reward:  {rewards.mean().item():.3f} ± {rewards.std().item():.3f}")
        print(f"   🎯 Post-training reward: {new_rewards.mean().item():.3f} ± {new_rewards.std().item():.3f}")
        print(f"   {improvement_emoji} Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        # Memory statistics
        final_stats = mann_model.get_memory_statistics()
        print(f"\n💾 Final Memory Statistics:")
        print(f"   📚 Total memories: {final_stats['total_memories']}")
        print(f"   📊 Memory utilization: {final_stats['memory_utilization']:.1%}")
        print(f"   🔍 Total retrievals: {final_stats['total_retrievals']}")
        print(f"   ✏️  Total writes: {final_stats['total_writes']}")
        print(f"   🧠 Memory matrix norm: {final_stats['memory_matrix_norm']:.4f}")
        
        print(f"\n✅ PPO Training Test Complete!")
        print(f"   📄 Detailed debug information: {debug_log_path}")
        print(f"   🎯 Training improved policy by {improvement_pct:+.1f}%")
        print(f"   📚 Used {final_stats['total_memories']} memories for training")
        
    except Exception as e:
        print(f"❌ PPO training test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ppo_training())