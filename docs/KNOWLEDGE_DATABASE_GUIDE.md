# Knowledge Database Integration Guide

## Tổng quan

Knowledge Database là hệ thống lưu trữ kiến thức được extract từ các cuộc hội thoại vào SQLite database. Hệ thống này tích hợp với Memory Manager để lưu trữ và quản lý knowledge một cách thông minh.

## Kiến trúc hệ thống

### Luồng xử lý
```
Cuộc hội thoại → LLM Extract → Memory Manager → Memory Operations → Knowledge Database
                     ↓                           ↓
            Extracted Knowledge              ADD/UPDATE/DELETE/NOOP
```

### Các thành phần chính

1. **KnowledgeDatabaseManager**: Quản lý SQLite database
2. **MemoryManager Integration**: Tích hợp với Memory Manager
3. **Memory Operations**: Tracking các operations (ADD/UPDATE/DELETE/NOOP)
4. **Knowledge Search**: Search và retrieval

## Database Schema

### Bảng `extracted_knowledge`
Lưu trữ kiến thức được extract từ conversations.

```sql
CREATE TABLE extracted_knowledge (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,              -- Original dialogue turn
    context TEXT,                       -- Conversation context
    entities TEXT,                      -- JSON array of entities
    intent TEXT,                        -- User intent
    key_facts TEXT,                     -- JSON array of key facts
    topics TEXT,                        -- JSON array of topics
    sentiment TEXT,                     -- Sentiment analysis
    importance REAL,                    -- Importance score (0-1)
    memory_type TEXT,                   -- Type: factual/personal/preference/procedure/other
    summary TEXT,                       -- Brief summary
    conversation_id TEXT,               -- Reference to conversation
    session_id TEXT,                    -- Reference to session
    timestamp TEXT,                     -- ISO timestamp
    metadata TEXT,                      -- JSON object with additional data
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Bảng `memory_operations`
Lưu trữ lịch sử các memory operations.

```sql
CREATE TABLE memory_operations (
    id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,       -- ADD/UPDATE/DELETE/NOOP
    target_memory_id TEXT,              -- ID của memory bị ảnh hưởng
    knowledge_id TEXT NOT NULL,         -- Reference to extracted_knowledge
    confidence REAL,                    -- Confidence score của operation
    reasoning TEXT,                     -- Lý do thực hiện operation
    execution_success BOOLEAN,          -- Kết quả thực hiện
    error_message TEXT,                 -- Error message nếu có
    timestamp TEXT,                     -- ISO timestamp
    metadata TEXT,                      -- JSON object với execution details
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (knowledge_id) REFERENCES extracted_knowledge (id)
);
```

### Bảng `conversation_turns`
Lưu trữ dialogue turns với references đến extracted knowledge.

```sql
CREATE TABLE conversation_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    conversation_id TEXT,
    turn_index INTEGER,
    user_message TEXT,
    bot_response TEXT,
    turn_context TEXT,
    knowledge_id TEXT,                  -- Reference to extracted knowledge
    timestamp TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (knowledge_id) REFERENCES extracted_knowledge (id)
);
```

## Cách sử dụng

### 1. Cấu hình trong RLChatbot

```python
config = {
    "enable_knowledge_db": True,
    "knowledge_db_path": "data/knowledge_bank.db",
    # ... other configs
}

agent = RLChatbotAgent(
    openai_model="gpt-4o-mini",
    api_key="your-api-key",
    config=config
)
```

### 2. Xử lý cuộc hội thoại

Khi xử lý message, hệ thống sẽ tự động:

1. Extract key information từ dialogue turn
2. Store extracted knowledge vào database
3. Quyết định memory operation (ADD/UPDATE/DELETE/NOOP)
4. Store memory operation record

```python
# Xử lý message tự động lưu vào Knowledge Database
result = agent.process_message(
    user_message="Làm thế nào để học Python?",
    context="Hướng dẫn lập trình"
)

# Kết quả sẽ include knowledge statistics
print(result["memory_stats"]["knowledge_database_stats"])
```

### 3. Truy vấn Knowledge Database

#### Search knowledge
```python
# Search trong session hiện tại
results = agent.search_knowledge(
    query="Python programming",
    memory_type="educational",
    min_importance=0.7
)

# Search across all sessions
results = agent.search_knowledge(
    query="machine learning",
    session_only=False
)
```

#### Lấy knowledge theo session
```python
knowledge = agent.get_knowledge_by_session(
    session_id="session_123",
    limit=20
)
```

#### Thống kê
```python
stats = agent.get_knowledge_statistics(days_back=7)
print(f"Total entries: {stats['total_knowledge_entries']}")
print(f"Operations: {stats['operations_stats']}")
```

### 4. Export data

```python
# Export knowledge của session
success = agent.export_session_knowledge(
    output_path="exports/session_knowledge.json",
    session_id="session_123"
)
```

## Memory Operations

Hệ thống sử dụng 4 loại memory operations:

### ADD
- **Khi nào**: Khi không có memory tương tự
- **Điều kiện**: Similarity < 0.8 và có đủ capacity
- **Kết quả**: Tạo memory entry mới

### UPDATE
- **Khi nào**: Khi có memory tương tự và có thông tin bổ sung
- **Điều kiện**: 0.8 ≤ Similarity < 0.95 và có new facts
- **Kết quả**: Merge thông tin vào memory existing

### DELETE
- **Khi nào**: Khi có memory rất tương tự và importance thấp hơn
- **Điều kiện**: Similarity ≥ 0.95 và importance mới ≤ existing
- **Kết quả**: Xóa memory redundant

### NOOP
- **Khi nào**: Khi không cần thay đổi gì
- **Điều kiện**: Below importance threshold hoặc memory đã đủ tốt
- **Kết quả**: Không thay đổi memory bank

## Configuration

### Cấu hình Memory Manager

```python
config = {
    # Knowledge Database
    "enable_knowledge_db": True,
    "knowledge_db_path": "data/knowledge_bank.db",
    
    # Memory Manager thresholds
    "similarity_threshold_update": 0.8,
    "similarity_threshold_delete": 0.95,
    "importance_threshold": 0.3,
    "max_memory_capacity": 5000,
    
    # LLM Extractor
    "llm_model": "gpt-4o-mini",
    "openai_api_key": "your-api-key"
}
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export KNOWLEDGE_DB_PATH="data/knowledge_bank.db"
```

## Monitoring và Analytics

### Performance Metrics

```python
insights = agent.get_memory_manager_insights()

# Operation statistics
stats = insights["operation_statistics"]
print(f"Total operations: {stats['total_operations']}")
print(f"ADD ratio: {stats['efficiency_metrics']['add_ratio']:.2%}")
print(f"UPDATE ratio: {stats['efficiency_metrics']['update_ratio']:.2%}")
print(f"NOOP ratio: {stats['efficiency_metrics']['noop_ratio']:.2%}")

# Knowledge Database stats
kb_stats = insights["knowledge_database"]
print(f"Total entries: {kb_stats['database_statistics']['total_knowledge_entries']}")
print(f"Average importance: {kb_stats['database_statistics']['avg_importance']:.2f}")
```

### Session Insights

```python
insights = agent.get_memory_manager_insights()
session_summary = insights["knowledge_database"]["session_summary"]

print(f"Session entries: {session_summary['total_entries']}")
print(f"Common topics: {session_summary['common_topics']}")
print(f"Memory types: {session_summary['memory_types']}")
```

## Maintenance

### Cleanup old data

```python
# Cleanup data older than 30 days
cleanup_results = agent.cleanup_old_data(days_threshold=30)
print(f"Cleaned up {cleanup_results['knowledge_cleaned']} knowledge entries")
print(f"Cleaned up {cleanup_results['operations_cleaned']} operations")
```

### Update daily statistics

```python
# Chạy hàng ngày để update statistics
agent.update_knowledge_daily_stats()
```

### Database maintenance

```python
# Get database stats
db_stats = agent.get_database_stats()
print(f"Database size: {db_stats.get('database_size_mb', 0):.2f} MB")

# Force save memory
agent.force_save_memory()
```

## Testing

### Chạy tests

```bash
# Test cơ bản (standalone)
python tests/test_knowledge_db_standalone.py

# Test tích hợp (cần API key)
python tests/test_knowledge_database.py
```

### Test results mẫu

```
🧪 Knowledge Database Standalone Test Suite
============================================================
Standalone Knowledge Database: ✅ PASSED
Database Performance: ✅ PASSED
Total: 2/2 tests passed
🎉 All standalone tests PASSED!
```

## Troubleshooting

### Common Issues

1. **Import errors**
   ```
   ModuleNotFoundError: No module named 'core'
   ```
   **Solution**: Sử dụng standalone test hoặc fix import paths

2. **Database lock errors**
   ```
   sqlite3.OperationalError: database is locked
   ```
   **Solution**: Đảm bảo đóng connections properly

3. **Memory operations failing**
   ```
   Failed to store memory operation
   ```
   **Solution**: Check database permissions và disk space

### Debug mode

```python
import logging
logging.getLogger("KnowledgeDB").setLevel(logging.DEBUG)
logging.getLogger("MemoryManager").setLevel(logging.DEBUG)
```

## Best Practices

1. **Regular cleanup**: Chạy cleanup_old_data() định kỳ
2. **Monitor performance**: Theo dõi operation ratios
3. **Backup database**: Backup knowledge_bank.db thường xuyên
4. **Optimize queries**: Sử dụng indexes cho large datasets
5. **Configuration tuning**: Điều chỉnh similarity thresholds theo use case

## API Reference

### KnowledgeDatabaseManager

```python
# Initialize
db = KnowledgeDatabaseManager("path/to/db.sqlite")

# Store knowledge
knowledge_id = db.store_extracted_knowledge(
    extracted_info=info_dict,
    dialogue_turn=turn_text,
    context=context_text,
    conversation_id=conv_id,
    session_id=session_id
)

# Store operation
operation_id = db.store_memory_operation(
    operation_type=MemoryOperation.ADD,
    knowledge_id=knowledge_id,
    confidence=0.9,
    reasoning="reason",
    execution_result={"success": True}
)

# Query
results = db.search_knowledge(
    query="search term",
    session_id="session_id",
    min_importance=0.5
)

# Statistics
stats = db.get_knowledge_statistics(days_back=7)
```

### RLChatbot Integration

```python
# Initialize with Knowledge DB
agent = RLChatbotAgent(config={"enable_knowledge_db": True})

# Use knowledge methods
results = agent.search_knowledge("query")
knowledge = agent.get_knowledge_by_session()
stats = agent.get_knowledge_statistics()
insights = agent.get_memory_manager_insights()
```

## Roadmap

- [ ] Full-text search với FTS5
- [ ] Vector similarity search
- [ ] Knowledge graph relationships
- [ ] Multi-user knowledge sharing
- [ ] Advanced analytics dashboard
- [ ] Export to external formats (CSV, Excel)
- [ ] API endpoints cho external access
