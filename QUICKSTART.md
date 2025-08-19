# 🚀 Quick Start Guide - RL Chatbot với OpenAI API

## Cài đặt nhanh (5 phút)

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup OpenAI API Key
```bash
# Tạo file .env hoặc export environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

# Hoặc tạo file .env
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

**💡 Lấy API key từ:** https://platform.openai.com/api-keys

### 3. Chạy demo
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

### 2. OpenAI Integration
- Sử dụng GPT-3.5-turbo hoặc GPT-4 để generate responses
- Kết hợp với RL memory system để cải thiện chất lượng
- Theo dõi token usage và chi phí API

### 3. Learning từ Feedback
- Đánh giá positive/negative cho responses
- Bot sẽ học và cải thiện theo thời gian
- RL system học từ feedback để optimize future responses

### 4. Memory Consolidation
- Sau ~50 interactions, system sẽ tự động consolidate memories
- Xem trong tab "🔍 Khám phá bộ nhớ"

### 5. Analytics Dashboard
- Real-time metrics
- Memory utilization
- Performance trends
- OpenAI API usage tracking

## 🔧 Cấu hình nhanh

Chỉnh sửa `configs/default.json`:

```json
{
  "openai_model": "gpt-3.5-turbo", // Hoặc "gpt-4" cho chất lượng tốt hơn
  "max_tokens": 150,                // Số tokens tối đa cho response
  "temperature": 0.8,               // Độ sáng tạo (0.1-2.0)
  "consolidation_threshold": 50,    // Số experiences để trigger consolidation
  "ewc_lambda": 1000.0,            // EWC regularization strength
  "decay_function": "exponential"   // Temporal decay function
}
```

### 🔑 OpenAI Models Available:
- **gpt-3.5-turbo**: Nhanh, rẻ, phù hợp cho development
- **gpt-4**: Chất lượng cao hơn, phù hợp cho production
- **gpt-4-turbo**: Cân bằng tốc độ và chất lượng

## 🐛 Troubleshooting

### OpenAI API Issues
```bash
# Kiểm tra API key
echo $OPENAI_API_KEY

# Test API connection
python -c "from openai import OpenAI; print('API key works!' if OpenAI().models.list() else 'Failed')"
```

### Common OpenAI Errors:
- **401 Unauthorized**: API key không hợp lệ
- **429 Rate Limit**: Vượt quá giới hạn request, đợi một chút
- **500 Server Error**: Lỗi từ OpenAI, thử lại sau

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

**Với OpenAI API:**
- Response time: ~500-2000ms (tùy thuộc vào OpenAI server)
- Memory usage: ~500MB-1GB RAM (ít hơn do không load LLM locally)
- Token usage: ~100-500 tokens per interaction
- Cost: ~$0.001-0.01 per interaction (tùy model)

**RL Training Performance:**
- Experience replay: ~50+ samples/second
- Memory consolidation: ~1-2 seconds for 100 experiences
- Neural network updates: ~10ms per batch

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

## 🧪 Testing with OpenAI

### Quick Test
```bash
# Test example
python examples/openai_usage_example.py

# Hoặc test trực tiếp
python -c "
from src.agents.rl_chatbot import RLChatbotAgent
import os
agent = RLChatbotAgent(api_key=os.getenv('OPENAI_API_KEY'))
agent.start_conversation()
result = agent.process_message('Xin chào!')
print(f'Bot: {result[\"response\"]}')
"
```

### Cost Estimation
- **Development**: ~$1-5/day với GPT-3.5-turbo
- **Production**: ~$10-50/day với GPT-4 (tùy traffic)

## 🆘 Support

Gặp vấn đề? Tạo issue với:
- OS version
- Python version  
- OpenAI API key status (hợp lệ/không)
- Error logs
- Steps to reproduce

**💡 Pro Tips:**
- Sử dụng GPT-3.5-turbo cho development để tiết kiệm cost
- Monitor token usage để tránh overspend
- Set usage limits trên OpenAI dashboard

Happy chatting! 🤖✨
