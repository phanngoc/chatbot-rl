# Chatbot Cá Nhân Hóa với Reinforcement Learning + OpenAI

Dự án MVP thiết kế chatbot cá nhân hóa sử dụng các thuật toán Reinforcement Learning, Episodic Memory tiên tiến, và tích hợp với OpenAI API để tạo ra trải nghiệm chat chất lượng cao.

## 🎯 Tính năng chính

### Thuật toán chính được triển khai:

1. **OpenAI Integration** - Sử dụng GPT-3.5-turbo/GPT-4 cho high-quality response generation
2. **Experience Replay (ER)** - Lưu trữ và replay các trải nghiệm để tránh catastrophic forgetting
3. **Retrieval-Augmented Episodic Memory** - Vector search với FAISS/ChromaDB cho memory retrieval
4. **Episodic Memory Consolidation** - Chuyển đổi từ episodic sang semantic memory (giống hippocampus)
5. **Elastic Weight Consolidation (EWC)** - Bảo vệ trọng số quan trọng khi học task mới
6. **Meta-learning với Episodic Memory** - MANN, NTM patterns để học cách chọn lọc trải nghiệm
7. **Temporal Decay & Importance Weighting** - Quản lý trọng số memory theo thời gian và importance

## 🚀 Cài đặt nhanh

```bash
# Clone repository
git clone <repository_url>
cd chatbot-rl

# Cài đặt dependencies
pip install -r requirements.txt

# Setup OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"
```

### 1. Interactive Chat (Terminal)
```bash
python src/main.py --mode interactive
```

### 2. Web Interface
```bash
streamlit run src/app.py
```

### 3. Training với data
```bash
python src/main.py --mode train --data examples/sample_training_data.json
```

### 4. Evaluation
```bash
python src/main.py --mode eval --data examples/sample_training_data.json
```

## 🧠 Kiến trúc hệ thống

### Experience Replay System
- Lưu trữ (state, action, reward, next_state) tuples
- Prioritized sampling dựa trên importance và recency
- Tự động lưu/load buffer

### Retrieval-Augmented Memory
- Vector search với FAISS hoặc ChromaDB
- Embedding-based similarity search
- Forgetting mechanism với temporal decay

### Memory Consolidation
- **Summarization**: LLM tóm tắt nhiều experiences với OpenAI API
- **Graph Integration**: Knowledge graph với concepts và relationships

### Elastic Weight Consolidation
- Fisher Information Matrix để xác định importance của weights
- Multi-task learning với penalty cho weight changes
- Adaptive lambda adjustment

### Meta-learning
- Memory-Augmented Neural Network (MANN)
- Episodic memory với attention mechanism
- Học cách select relevant memories

### Temporal Weighting
- Multiple decay functions (exponential, power-law, forgetting curve)
- Importance weighting dựa trên feedback và usage
- Access pattern analysis

## 🌟 Hệ thống Feedback

Chatbot sử dụng hệ thống đánh giá **1-5 sao** cho mỗi phản hồi:

- **⭐1**: Rất kém (feedback score: -1.0)
- **⭐2**: Kém (feedback score: -0.5)
- **⭐3**: Trung bình (feedback score: 0.0)
- **⭐4**: Tốt (feedback score: 0.5)
- **⭐5**: Rất tốt (feedback score: 1.0)

### Cách sử dụng:
- **Web Interface**: Nhấn vào các nút ⭐1-5 sau mỗi phản hồi
- **Terminal**: Nhập số từ 1-5 khi được hỏi feedback
- **Batch Training**: Score tự động dựa trên similarity với expected response

MIT License - xem file LICENSE để biết chi tiết.

## 🙏 Acknowledgments

- Transformers library từ Hugging Face
- FAISS cho vector search
- ChromaDB cho vector database
- Streamlit cho web interface
