#!/usr/bin/env python3
"""
Demo Script cho Knowledge Distillation System
Thể hiện cách sử dụng Teacher-Student architecture đúng chuẩn
"""

import os
import sys
sys.path.append('src')

from memory.consolidation import ModelDistillation
from datetime import datetime
import json
import torch

def create_sample_memories():
    """Tạo sample episodic memories cho demo"""
    memories = [
        {
            "id": "mem_001",
            "context": "User hỏi về machine learning",
            "content": "Machine learning là gì? Tôi muốn học về neural networks và deep learning.",
            "reward": 0.9,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_002", 
            "context": "User quan tâm về Python programming",
            "content": "Làm thế nào để code Python hiệu quả? Tôi muốn tìm hiểu về best practices.",
            "reward": 0.8,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_003",
            "context": "User hỏi về data science",
            "content": "Data science workflow như thế nào? Từ data collection đến model deployment.",
            "reward": 0.85,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_004",
            "context": "User quan tâm AI ethics",
            "content": "Tại sao AI ethics lại quan trọng? Bias trong machine learning có nghiêm trọng không?",
            "reward": 0.7,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_005",
            "context": "User hỏi về career development",
            "content": "Làm thế nào để trở thành AI engineer? Roadmap học tập như thế nào?",
            "reward": 0.75,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_006",
            "context": "User hỏi về computer vision",
            "content": "Computer vision có ứng dụng gì trong thực tế? CNN hoạt động như thế nào?",
            "reward": 0.9,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_007",
            "context": "User quan tâm NLP",
            "content": "Natural Language Processing khó không? Transformer architecture thế nào?",
            "reward": 0.85,
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "mem_008",
            "context": "User hỏi về deployment",
            "content": "Deploy model ML như thế nào? Docker, Kubernetes có cần thiết không?",
            "reward": 0.8,
            "timestamp": datetime.now().isoformat()
        }
    ]
    return memories

def demo_knowledge_distillation():
    """Demo chính cho Knowledge Distillation"""
    print("🎓 DEMO: Knowledge Distillation cho Episodic Memory System")
    print("=" * 60)
    
    # Kiểm tra OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Cần OPENAI_API_KEY để chạy demo này!")
        return
    
    try:
        # 1. Khởi tạo ModelDistillation với parameters tối ưu
        print("\n1️⃣ Khởi tạo Knowledge Distillation System...")
        distillation = ModelDistillation(
            learning_rate=1e-4,
            temperature=4.0,      # Temperature scaling cho soft targets
            alpha=0.7,           # Weight cho distillation loss
            beta=0.3,            # Weight cho student loss  
            openai_model="gpt-4o-mini"
        )
        
        print(f"✅ Distillation system initialized")
        print(f"   - Temperature: {distillation.temperature}")
        print(f"   - Alpha (distillation): {distillation.alpha}")
        print(f"   - Beta (student): {distillation.beta}")
        print(f"   - Device: {distillation.device}")
        
        # 2. Tạo sample memories
        print("\n2️⃣ Tạo sample episodic memories...")
        memories = create_sample_memories()
        print(f"✅ Created {len(memories)} sample memories")
        
        # 3. Chạy Knowledge Distillation
        print("\n3️⃣ Bắt đầu Knowledge Distillation Process...")
        print("   Phase 1: Training Teacher Model...")
        print("   Phase 2: Distilling to Student Model...")
        
        results = distillation.distill_from_memories(
            memories=memories,
            num_epochs=3,        # Reduce epochs cho demo
            batch_size=4
        )
        
        print(f"\n✅ Knowledge Distillation completed!")
        print(f"   Status: {results['status']}")
        
        if results['status'] == 'completed':
            # In kết quả chi tiết
            teacher_results = results['teacher_training']
            distill_results = results['distillation']
            
            print(f"\n📊 Teacher Training Results:")
            print(f"   - Success: {teacher_results['success']}")
            print(f"   - Average Loss: {teacher_results['avg_loss']:.4f}")
            print(f"   - Epochs: {teacher_results['epochs']}")
            print(f"   - Batches: {teacher_results['batches']}")
            
            print(f"\n📊 Distillation Results:")
            print(f"   - Success: {distill_results['success']}")
            print(f"   - Distillation Loss: {distill_results['avg_distillation_loss']:.4f}")
            print(f"   - Student Loss: {distill_results['avg_student_loss']:.4f}")
            print(f"   - Epochs: {distill_results['epochs']}")
            print(f"   - Batches: {distill_results['batches']}")
            
            print(f"\n🎯 Distillation Parameters:")
            params = results['parameters']
            print(f"   - Temperature: {params['temperature']}")
            print(f"   - Alpha: {params['alpha']}")
            print(f"   - Beta: {params['beta']}")
        
        # 4. Test Student Model
        print("\n4️⃣ Testing Student Model...")
        test_queries = [
            "Tôi muốn học về deep learning và neural networks",
            "Python programming best practices là gì?",
            "Computer vision có ứng dụng gì?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Test {i}: {query}")
            prediction = distillation.get_student_prediction(query)
            
            if "error" not in prediction:
                print(f"   → Prediction Class: {prediction['prediction_class']}")
                print(f"   → Confidence: {prediction['confidence']:.4f}")
                print(f"   → Model Type: {prediction['model_type']}")
                print(f"   → Model Size: {prediction['compressed_size']}")
            else:
                print(f"   → Error: {prediction['error']}")
        
        # 5. So sánh Teacher vs Student
        print("\n5️⃣ Comparing Teacher vs Student Models...")
        comparison_query = "Machine learning và AI có gì khác nhau?"
        
        comparison = distillation.compare_teacher_student(comparison_query)
        
        if "error" not in comparison:
            print(f"\n   Query: {comparison_query}")
            print(f"\n   👨‍🏫 Teacher Model:")
            print(f"      - Prediction: {comparison['teacher']['prediction_class']}")
            print(f"      - Confidence: {comparison['teacher']['confidence']:.4f}")
            print(f"      - Size: {comparison['teacher']['model_size']}")
            
            print(f"\n   🎒 Student Model:")
            print(f"      - Prediction: {comparison['student']['prediction_class']}")
            print(f"      - Confidence: {comparison['student']['confidence']:.4f}")
            print(f"      - Size: {comparison['student']['model_size']}")
            
            print(f"\n   📈 Similarity Analysis:")
            similarity = comparison['similarity']
            print(f"      - Predictions Match: {similarity['prediction_match']}")
            print(f"      - KL Divergence: {similarity['kl_divergence']:.4f}")
            print(f"      - Confidence Diff: {similarity['confidence_diff']:.4f}")
            print(f"      - Compression Ratio: {comparison['compression_ratio']:.2f}x")
        
        # 6. Lưu models
        print("\n6️⃣ Saving Trained Models...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results = distillation.save_models(f"models/distillation_{timestamp}")
        
        print(f"✅ Models saved:")
        print(f"   - Teacher: {save_results['teacher_model']}")
        print(f"   - Student: {save_results['student_model']}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Insights:")
        print("   • Teacher model học patterns từ episodic memories")
        print("   • Student model học từ Teacher qua soft targets")
        print("   • Temperature scaling làm mềm probability distributions")
        print("   • KL Divergence measure sự tương đồng giữa models")
        print("   • Student model nhỏ hơn nhưng vẫn giữ được knowledge")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_knowledge_distillation()
