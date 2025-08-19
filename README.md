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

# Chạy demo
./run_demo.sh
```

**📖 Chi tiết:** Xem [QUICKSTART.md](QUICKSTART.md) để hướng dẫn setup và sử dụng OpenAI API

## 📁 Cấu trúc dự án

```
chatbot-rl/
├── src/
│   ├── core/                    # Core RL algorithms
│   │   ├── experience_replay.py # Experience Replay system
│   │   ├── ewc.py              # Elastic Weight Consolidation
│   │   ├── meta_learning.py    # Meta-learning & MANN
│   │   └── temporal_weighting.py # Temporal decay & weighting
│   ├── memory/                  # Memory systems
│   │   ├── retrieval_memory.py  # Vector-based memory retrieval
│   │   └── consolidation.py     # Memory consolidation
│   ├── agents/                  # Chatbot agents
│   │   └── rl_chatbot.py       # Main RL chatbot agent
│   ├── main.py                 # CLI application
│   └── app.py                  # Streamlit web interface
├── data/                       # Data storage
├── configs/                    # Configuration files
├── examples/                   # Sample data
└── run_demo.sh                # Demo script
```

## 🎮 Sử dụng

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

### 5. Demo script
```bash
./run_demo.sh
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
- **Summarization**: LLM tóm tắt nhiều experiences
- **Graph Integration**: Knowledge graph với concepts
- **Distillation**: Fine-tune model weights từ episodic memories

### Elastic Weight Consolidation
- Fisher Information Matrix để xác định importance của weights
- Multi-task learning với penalty cho weight changes
- Adaptive lambda adjustment

### Meta-learning
- Memory-Augmented Neural Network (MANN)
- Neural Turing Machine (NTM) patterns
- Học cách select relevant memories

### Temporal Weighting
- Multiple decay functions (exponential, power-law, forgetting curve)
- Importance weighting dựa trên feedback và usage
- Access pattern analysis

## ⚙️ Cấu hình

Chỉnh sửa `configs/default.json`:

```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "device": "cpu",
  "experience_buffer_size": 10000,
  "memory_store_type": "chroma",
  "max_memories": 5000,
  "consolidation_threshold": 100,
  "ewc_lambda": 1000.0,
  "decay_function": "exponential",
  "temperature": 0.8
}
```

## 📊 Web Interface Features

- **💬 Chat Interface**: Trò chuyện real-time với bot
- **📊 Analytics Dashboard**: Metrics và performance tracking
- **🔍 Memory Explorer**: Khám phá và search memories
- **⚙️ Settings**: Cấu hình system parameters

## 🔧 API Usage

```python
from src.agents.rl_chatbot import RLChatbotAgent

# Initialize agent
agent = RLChatbotAgent(config={
    "temperature": 0.8,
    "ewc_lambda": 1000.0
})

# Start conversation
conversation_id = agent.start_conversation()

# Process message
result = agent.process_message("Xin chào!")
print(result['response'])

# Provide feedback
agent.provide_feedback(result['experience_id'], 0.8)

# Save/load state
agent.save_agent_state("data/my_agent.json")
agent.load_agent_state("data/my_agent.json")
```

## 📈 Performance Metrics

System theo dõi các metrics sau:
- Total interactions
- Positive/negative feedback ratio
- Average response time
- Memory utilization
- Consolidation frequency
- Meta-learning episodes

## 🛠️ Development

### Chạy tests
```bash
python -m pytest tests/
```

### Thêm thuật toán mới
1. Tạo module trong `src/core/`
2. Implement interface tương thích
3. Tích hợp vào `RLChatbotAgent`
4. Update configuration

### Custom Memory Store
```python
from src.memory.retrieval_memory import RetrievalAugmentedMemory

# Sử dụng FAISS
memory = RetrievalAugmentedMemory(store_type="faiss")

# Hoặc ChromaDB
memory = RetrievalAugmentedMemory(store_type="chroma")
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết chi tiết.

## 🙏 Acknowledgments

- Transformers library từ Hugging Face
- FAISS cho vector search
- ChromaDB cho vector database
- Streamlit cho web interface
