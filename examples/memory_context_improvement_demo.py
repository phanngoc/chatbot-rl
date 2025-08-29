#!/usr/bin/env python3
"""
Demo Script: Cải thiện hàm _format_memory_context
Minh họa sự khác biệt giữa phiên bản cũ và mới
"""

import torch
import numpy as np
from typing import List, Dict, Any

class MockRLChatbotModel:
    """Mock model để demo _format_memory_context improvements"""
    
    def __init__(self):
        self.memory_dim = 256
        
    def _format_memory_context_old(self, memory_context: torch.Tensor) -> str:
        """Phiên bản cũ - chỉ trả về số lượng memories"""
        if memory_context is None:
            return ""
        
        try:
            if memory_context.dim() == 3:
                batch_size, num_memories, memory_dim = memory_context.shape
                return f"Có {num_memories} memories liên quan được tìm thấy từ các cuộc hội thoại trước."
            elif memory_context.dim() == 2:
                num_memories, memory_dim = memory_context.shape
                return f"Có {num_memories} memories liên quan được tìm thấy từ các cuộc hội thoại trước."
            else:
                return "Có memories liên quan được tìm thấy từ các cuộc hội thoại trước."
        except Exception as e:
            return "Có memories liên quan được tìm thấy từ các cuộc hội thoại trước."
    
    def _format_memory_context_new(self, memory_context: torch.Tensor, retrieved_info: List[Dict] = None) -> str:
        """Phiên bản mới - cung cấp thông tin chi tiết"""
        if memory_context is None:
            return ""
        
        try:
            # Validate tensor shape and dimensions
            num_memories = 0
            memory_dim = 0
            
            if memory_context.dim() == 3:
                batch_size, num_memories, memory_dim = memory_context.shape
            elif memory_context.dim() == 2:
                num_memories, memory_dim = memory_context.shape
            else:
                return "Có thông tin memory liên quan từ các cuộc hội thoại trước."
            
            if num_memories == 0:
                return ""
                
            # Khởi tạo thông tin cơ bản
            context_parts = [f"📚 Tìm thấy {num_memories} memories liên quan từ {memory_dim}D memory space."]
            
            # Nếu có retrieved_info, thêm thông tin chi tiết
            if retrieved_info and len(retrieved_info) > 0:
                # Phân tích thông tin memories
                total_similarity = 0
                total_importance = 0
                total_usage = 0
                high_quality_memories = 0
                
                memory_details = []
                
                for i, info in enumerate(retrieved_info[:3]):  # Top 3 memories
                    if isinstance(info, dict):
                        similarity = info.get('similarity', 0)
                        importance = info.get('importance_weight', 1.0)
                        usage = info.get('usage_count', 0)
                        
                        total_similarity += similarity
                        total_importance += importance
                        total_usage += usage
                        
                        # Đánh giá chất lượng memory
                        if similarity > 0.7 and importance > 1.2:
                            high_quality_memories += 1
                        
                        # Format thông tin memory
                        quality_indicator = "🔥" if similarity > 0.8 else "⭐" if similarity > 0.6 else "💡"
                        memory_details.append(
                            f"  {quality_indicator} Memory #{i+1}: "
                            f"độ liên quan {similarity:.1%}, "
                            f"quan trọng {importance:.1f}x, "
                            f"đã dùng {usage} lần"
                        )
                
                # Tính toán thống kê tổng thể
                if num_memories > 0:
                    avg_similarity = total_similarity / min(len(retrieved_info), num_memories)
                    avg_importance = total_importance / min(len(retrieved_info), num_memories)
                    avg_usage = total_usage / min(len(retrieved_info), num_memories)
                    
                    # Thêm thông tin chất lượng tổng thể
                    quality_summary = f"📊 Chất lượng memories: độ liên quan trung bình {avg_similarity:.1%}"
                    
                    if high_quality_memories > 0:
                        quality_summary += f", có {high_quality_memories} memories chất lượng cao"
                    
                    if avg_importance > 1.3:
                        quality_summary += f", mức độ quan trọng cao ({avg_importance:.1f}x)"
                    elif avg_importance < 0.8:
                        quality_summary += f", mức độ quan trọng thấp ({avg_importance:.1f}x)"
                    
                    if avg_usage > 5:
                        quality_summary += f", được sử dụng thường xuyên ({avg_usage:.0f} lần TB)"
                    
                    context_parts.append(quality_summary)
                
                # Thêm chi tiết memories
                if memory_details:
                    context_parts.extend(memory_details)
                
                # Phân tích utilization và fragmentation
                memory_utilization = min(len(retrieved_info), num_memories) / max(num_memories, 1)
                if memory_utilization < 0.5:
                    context_parts.append(f"⚠️  Memory utilization thấp: {memory_utilization:.1%}")
                
                # Đánh giá hiệu quả memory
                if avg_similarity > 0.8 and high_quality_memories >= 2:
                    context_parts.append("✅ Memory system hoạt động hiệu quả với memories chất lượng cao")
                elif avg_similarity < 0.4:
                    context_parts.append("⚠️  Cần cải thiện: memories có độ liên quan thấp")
                
            else:
                # Thông tin cơ bản khi không có retrieved_info
                estimated_utilization = min(num_memories / 100.0, 1.0)  # Giả sử max 100 memories
                context_parts.append(f"📈 Estimated memory utilization: {estimated_utilization:.1%}")
                
                if memory_dim != self.memory_dim:
                    context_parts.append(f"⚠️  Memory dimension mismatch: expected {self.memory_dim}D, got {memory_dim}D")
            
            return "\n".join(context_parts)
                
        except Exception as e:
            return f"Có {getattr(memory_context, 'size', lambda: [0])(0) if hasattr(memory_context, 'size') else 'một số'} memories liên quan từ các cuộc hội thoại trước."


def create_sample_data():
    """Tạo sample data để demo"""
    
    # Tạo memory context tensor
    memory_context = torch.randn(1, 4, 256)  # 4 memories, 256 dimensions
    
    # Tạo retrieved_info với thông tin chi tiết
    retrieved_info = [
        {
            "index": 0,
            "similarity": 0.85,
            "importance_weight": 1.5,
            "usage_count": 3
        },
        {
            "index": 1,
            "similarity": 0.72,
            "importance_weight": 1.8,
            "usage_count": 7
        },
        {
            "index": 2,
            "similarity": 0.45,
            "importance_weight": 0.9,
            "usage_count": 1
        },
        {
            "index": 3,
            "similarity": 0.91,
            "importance_weight": 2.0,
            "usage_count": 12
        }
    ]
    
    return memory_context, retrieved_info


def main():
    """Chạy demo comparison"""
    
    print("=" * 80)
    print("DEMO: Cải thiện hàm _format_memory_context")
    print("=" * 80)
    
    # Khởi tạo mock model
    model = MockRLChatbotModel()
    
    # Tạo sample data
    memory_context, retrieved_info = create_sample_data()
    
    print("\n📊 Sample Data:")
    print(f"Memory context shape: {memory_context.shape}")
    print(f"Retrieved info: {len(retrieved_info)} memories")
    for i, info in enumerate(retrieved_info):
        print(f"  Memory {i+1}: similarity={info['similarity']:.2f}, "
              f"importance={info['importance_weight']:.1f}x, "
              f"usage={info['usage_count']} lần")
    
    print("\n" + "="*50)
    print("🔴 PHIÊN BẢN CŨ:")
    print("="*50)
    old_result = model._format_memory_context_old(memory_context)
    print(old_result)
    
    print("\n" + "="*50)
    print("🟢 PHIÊN BẢN MỚI:")
    print("="*50)
    new_result = model._format_memory_context_new(memory_context, retrieved_info)
    print(new_result)
    
    print("\n" + "="*80)
    print("📈 PHÂN TÍCH SỰ KHÁC BIỆT:")
    print("="*80)
    
    print("🔴 Phiên bản cũ:")
    print("  ❌ Chỉ cung cấp số lượng memories")
    print("  ❌ Không có thông tin về chất lượng")
    print("  ❌ Không có thông tin về utilization")
    print("  ❌ Không hỗ trợ debugging và optimization")
    
    print("\n🟢 Phiên bản mới:")
    print("  ✅ Cung cấp thông tin chi tiết về từng memory")
    print("  ✅ Phân tích chất lượng và độ liên quan")
    print("  ✅ Hiển thị usage patterns và importance weights")
    print("  ✅ Đánh giá memory utilization và fragmentation")
    print("  ✅ Cảnh báo vấn đề và gợi ý cải thiện")
    print("  ✅ Hỗ trợ debugging và performance monitoring")
    
    print("\n📝 KẾT LUẬN:")
    print("Phiên bản mới cung cấp cái nhìn toàn diện về memory system,")
    print("giúp chẩn đoán hiệu quả và tối ưu hóa hiệu suất chatbot.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
