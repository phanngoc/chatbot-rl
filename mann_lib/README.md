# MANN CLI Chatbot

Memory-Augmented Neural Network (MANN) CLI Chatbot - Hệ thống chatbot dòng lệnh với khả năng ghi nhớ và học tập thông minh.

## 🚀 Tính Năng

- **Memory-Augmented Neural Network**: Sử dụng external memory để lưu trữ và truy xuất thông tin
- **CLI Interface**: Giao diện dòng lệnh thân thiện với người dùng
- **RESTful API**: API server để tích hợp với các ứng dụng khác
- **Production Monitoring**: Hệ thống giám sát và cảnh báo cho môi trường production
- **Pager System**: Tích hợp hệ thống pager để cảnh báo khi có sự cố
- **Health Checks**: Kiểm tra sức khỏe hệ thống tự động

## 📁 Cấu Trúc Thư Mục

```
cli/
├── standalone_mann/          # Thư viện MANN độc lập
│   ├── __init__.py
│   ├── mann_core.py         # Core MANN implementation
│   ├── mann_config.py       # Configuration management
│   ├── mann_api.py          # RESTful API server
│   └── mann_monitoring.py   # Monitoring và pager system
├── mann_chatbot.py          # CLI chatbot chính
├── run_chatbot.py           # Script chạy chatbot
├── run_api.py               # Script chạy API server
├── test_mann.py             # Test suite
├── requirements.txt         # Dependencies
└── README.md               # Documentation này
```

## 🛠️ Cài Đặt

### 1. Cài đặt Dependencies

```bash
cd cli
pip install -r requirements.txt
```

### 2. Cấu hình Environment Variables (Optional)

```bash
export MANN_INPUT_SIZE=768
export MANN_HIDDEN_SIZE=256
export MANN_MEMORY_SIZE=1000
export MANN_API_HOST=localhost
export MANN_API_PORT=8000
export MANN_LOG_LEVEL=INFO
export MANN_ENABLE_PAGER=true
export MANN_PAGER_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## 🚀 Sử Dụng

### 1. Chạy CLI Chatbot

```bash
# Chạy chatbot interactive mode
python run_chatbot.py

# Hoặc chạy trực tiếp
python mann_chatbot.py
```

### 2. Chạy API Server

```bash
# Chạy API server
python run_api.py

# API sẽ chạy tại http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### 3. Test Hệ Thống

```bash
# Chạy test suite
python test_mann.py
```

## 💬 CLI Commands

Khi chạy chatbot, bạn có thể sử dụng các lệnh sau:

- `/help` - Hiển thị trợ giúp
- `/stats` - Hiển thị thống kê chatbot
- `/search <query>` - Tìm kiếm memories
- `/health` - Kiểm tra sức khỏe hệ thống
- `/quit` - Thoát chatbot

## 🌐 API Endpoints

### Health Check
```http
GET /health
```

### Add Memory
```http
POST /memories
Content-Type: application/json

{
  "content": "Memory content",
  "context": "Context information",
  "tags": ["tag1", "tag2"],
  "importance_weight": 1.5,
  "metadata": {"key": "value"}
}
```

### Search Memories
```http
POST /search
Content-Type: application/json

{
  "query": "search query",
  "top_k": 5,
  "min_similarity": 0.0
}
```

### Process Query
```http
POST /query
Content-Type: application/json

{
  "input_text": "User input",
  "retrieve_memories": true
}
```

### Get Statistics
```http
GET /statistics
```

## 📊 Monitoring và Pager System

### 1. Monitoring Features

- **Performance Metrics**: Theo dõi thời gian xử lý, memory utilization
- **System Resources**: CPU, RAM, Disk usage
- **Error Tracking**: Đếm và theo dõi lỗi
- **Query Throughput**: Số lượng queries per minute

### 2. Pager System

Hệ thống pager tự động gửi cảnh báo khi:

- Memory utilization > 90%
- Processing time > 5 seconds
- Error rate > 10%
- CPU usage > 80%
- System memory usage > 80%
- Disk usage > 90%

### 3. Alert Levels

- **INFO**: Thông tin chung
- **WARNING**: Cảnh báo nhẹ
- **ERROR**: Lỗi nghiêm trọng
- **CRITICAL**: Khẩn cấp

### 4. Webhook Integration

Cấu hình webhook để nhận cảnh báo:

```bash
export MANN_PAGER_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## 🔧 Configuration

### MANNConfig Options

```python
# Model parameters
input_size: int = 768
hidden_size: int = 256
memory_size: int = 1000
memory_dim: int = 128
output_size: int = 768

# Memory management
similarity_threshold_update: float = 0.8
similarity_threshold_delete: float = 0.95
importance_threshold: float = 0.3
max_memory_capacity: int = 5000

# API settings
api_host: str = "localhost"
api_port: int = 8000
api_timeout: int = 30

# Monitoring
enable_monitoring: bool = True
enable_pager: bool = True
health_check_interval: int = 60  # seconds

# Storage
data_dir: str = "./data"
model_save_path: str = "./models/mann_model.pt"
memory_save_path: str = "./data/memory_bank.pkl"
```

## 🏗️ Production Deployment

### 1. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_api.py"]
```

### 2. Systemd Service

```ini
[Unit]
Description=MANN API Server
After=network.target

[Service]
Type=simple
User=mann
WorkingDirectory=/opt/mann
ExecStart=/opt/mann/venv/bin/python run_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Health Check Script

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ MANN API is healthy"
    exit 0
else
    echo "❌ MANN API is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

## 🧪 Testing

### 1. Unit Tests

```bash
# Chạy test suite
python test_mann.py

# Test specific components
python -m pytest tests/test_mann_core.py
python -m pytest tests/test_mann_api.py
```

### 2. Load Testing

```bash
# Sử dụng Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Sử dụng wrk
wrk -t12 -c400 -d30s http://localhost:8000/health
```

### 3. Memory Testing

```bash
# Test memory operations
python -c "
from standalone_mann.mann_core import MemoryAugmentedNetwork
from standalone_mann.mann_config import MANNConfig

config = MANNConfig()
mann = MemoryAugmentedNetwork(**config.__dict__)

# Add 1000 memories
for i in range(1000):
    mann.add_memory(f'Memory {i}', f'Context {i}', [f'tag{i}'])

print(f'Total memories: {len(mann.memory_bank)}')
print(f'Memory utilization: {mann.get_memory_statistics()[\"memory_utilization\"]:.2%}')
"
```

## 🔍 Troubleshooting

### 1. Common Issues

**Memory Bank Not Loading**
```bash
# Check file permissions
ls -la data/memory_bank.pkl

# Check file format
python -c "import pickle; print(pickle.load(open('data/memory_bank.pkl', 'rb')))"
```

**API Server Not Starting**
```bash
# Check port availability
netstat -tulpn | grep :8000

# Check logs
tail -f logs/mann.log
```

**High Memory Usage**
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

### 2. Performance Optimization

**Reduce Memory Size**
```python
config = MANNConfig()
config.memory_size = 500  # Reduce from 1000
config.max_memory_capacity = 2500  # Reduce from 5000
```

**Enable Caching**
```python
config.enable_caching = True
config.cache_size = 1000
config.cache_ttl = 3600
```

**Adjust Monitoring Interval**
```python
config.health_check_interval = 120  # Increase from 60 seconds
```

## 📈 Performance Metrics

### 1. Expected Performance

- **Memory Retrieval**: < 100ms
- **Memory Storage**: < 50ms
- **Query Processing**: < 500ms
- **API Response Time**: < 200ms
- **Memory Utilization**: < 80%

### 2. Scaling Guidelines

- **Memory Size**: 1000 memories = ~100MB RAM
- **Concurrent Users**: 100 users = ~1GB RAM
- **API Throughput**: 1000 requests/minute
- **Storage**: 1GB for 10,000 memories

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Memory-Augmented Neural Networks research
- FastAPI framework
- PyTorch deep learning library
- Production monitoring best practices
