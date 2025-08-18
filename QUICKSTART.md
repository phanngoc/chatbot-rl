# 🚀 Quick Start Guide - RL Chatbot

## Cài đặt nhanh (5 phút)

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy demo
```bash
./run_demo.sh
```

Chọn option 2 để chạy Web Interface!

## 🎮 Demo Web Interface

1. **Mở trình duyệt**: http://localhost:8501
2. **Bắt đầu trò chuyện**: Nhập tin nhắn vào chat box
3. **Đánh giá phản hồi**: Nhấn 👍 hoặc 👎 để cung cấp feedback
4. **Xem phân tích**: Chuyển sang tab "📊 Phân tích"

## 🧪 Test với sample data

```bash
# Training
python src/main.py --mode train --data examples/sample_training_data.json

# Evaluation  
python src/main.py --mode eval --data examples/sample_training_data.json
```

## 🎯 Tính năng nổi bật để test

### 1. Memory Retrieval
- Hỏi về chủ đề đã nói trước đó
- Bot sẽ sử dụng memories để trả lời nhất quán

### 2. Learning từ Feedback
- Đánh giá positive/negative cho responses
- Bot sẽ học và cải thiện theo thời gian

### 3. Memory Consolidation
- Sau ~50 interactions, system sẽ tự động consolidate memories
- Xem trong tab "🔍 Khám phá bộ nhớ"

### 4. Analytics Dashboard
- Real-time metrics
- Memory utilization
- Performance trends

## 🔧 Cấu hình nhanh

Chỉnh sửa `configs/default.json`:

```json
{
  "temperature": 0.8,           // Độ sáng tạo (0.1-2.0)
  "consolidation_threshold": 50, // Số experiences để trigger consolidation
  "ewc_lambda": 1000.0,         // EWC regularization strength
  "decay_function": "exponential" // Temporal decay function
}
```

## 🐛 Troubleshooting

### Lỗi import modules
```bash
cd src
export PYTHONPATH=$PYTHONPATH:$(pwd)
python main.py
```

### Lỗi ChromaDB
- ChromaDB sẽ tự tạo database trong `data/chroma_db/`
- Xóa folder này nếu gặp lỗi và restart

### Memory usage cao
- Giảm `experience_buffer_size` trong config
- Chạy cleanup: Tab Settings -> "🧹 Dọn dẹp memories cũ"

## 📊 Expected Performance

**Trên CPU (MacBook Pro M1):**
- Response time: ~200-500ms
- Memory usage: ~1-2GB RAM
- Training: ~10 samples/second

**Trên GPU:**
- Response time: ~100-200ms  
- Training: ~50+ samples/second

## 🎪 Demo Scenarios

### Scenario 1: Consistent Personality
1. "Tên bạn là gì?" 
2. Chat về nhiều chủ đề khác
3. "Bạn nhớ tên mình không?" -> Bot nhớ và trả lời nhất quán

### Scenario 2: Learning from Feedback
1. Hỏi câu hỏi về lập trình
2. Đánh giá negative nếu response không tốt
3. Hỏi câu tương tự -> Bot sẽ cải thiện

### Scenario 3: Memory Consolidation
1. Chat 50+ tin nhắn về các chủ đề khác nhau
2. Xem tab "📊 Phân tích" -> Consolidation runs tăng
3. Tab "🔍 Khám phá bộ nhớ" -> Có consolidated knowledge

## 💡 Tips

- **Feedback thường xuyên**: Giúp bot học nhanh hơn
- **Chủ đề đa dạng**: Test khả năng memory retrieval
- **Monitor analytics**: Theo dõi performance metrics
- **Save state**: Lưu trạng thái để tiếp tục sau

## 🆘 Support

Gặp vấn đề? Tạo issue với:
- OS version
- Python version  
- Error logs
- Steps to reproduce

Happy chatting! 🤖✨
