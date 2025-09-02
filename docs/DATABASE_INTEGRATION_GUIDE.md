# Database Integration Guide

## Tổng quan

Hệ thống RL Chatbot đã được nâng cấp với database persistence hoàn chỉnh để lưu trữ **memory bank** và **session chat history**. Điều này đảm bảo rằng:

- ✅ **Memory bank** được lưu trữ persistent và restore chính xác theo session
- ✅ **Chat history** được mapping với sessions và có thể restore khi cần
- ✅ **Session management** hoàn chỉnh với database SQLite
- ✅ **Auto-save** memory bank trong quá trình conversation
- ✅ **Migration tool** để chuyển đổi dữ liệu cũ

## Kiến trúc hệ thống

### Database Schema

```sql
-- Chat Sessions
chat_sessions (
    session_id, created_at, last_updated, total_messages, metadata
)

-- Chat Messages  
chat_messages (
    message_id, session_id, role, content, timestamp, metadata
)

-- Memory Bank States (full backup)
memory_bank_states (
    session_id, memory_entries_blob, timestep, created_at, last_updated
)

-- Memory Bank Entries (queryable)
memory_bank_entries (
    entry_id, session_id, key_vector, value_vector, 
    usage_count, last_accessed, importance_weight, created_at, updated_at
)
```

### Các thành phần chính

1. **DatabaseManager**: Quản lý SQLite database operations
2. **SessionManager**: Quản lý session lifecycle và data operations  
3. **MetaLearningEpisodicSystem**: Tích hợp với database persistence
4. **RLChatbotAgent**: Session-aware agent với auto-save
5. **DataMigrationTool**: Tool để migrate existing data

## Hướng dẫn sử dụng

### 1. Khởi tạo hệ thống mới

```python
from agents.rl_chatbot import RLChatbotAgent

# Tạo agent
agent = RLChatbotAgent(config={
    "temperature": 0.8,
    "max_tokens": 150
})

# Bắt đầu session mới
session_id = agent.start_session(user_id="user123")
print(f"Started session: {session_id}")
```

### 2. Chat và auto-save memory

```python
# Process messages - memory tự động được save
result = agent.process_message("Hello, how are you?")
print(result['response'])

# Memory bank được auto-save mỗi 5 interactions
# Session data được save ngay lập tức
```

### 3. Resume existing session

```python
# List recent sessions
sessions = agent.list_recent_sessions(10)
for session in sessions:
    print(f"Session {session['session_id']}: {session['total_messages']} messages")

# Resume session cụ thể
success = agent.resume_session(session_id)
if success:
    print("Session resumed successfully")
    
    # Chat history được restore tự động
    result = agent.process_message("Continue our conversation")
```

### 4. Session management

```python
# Get session summary
summary = agent.get_session_summary()
print(f"Session has {summary['total_messages']} messages")
print(f"Memory bank size: {summary['memory_stats']['total_entries']}")

# Force save memory bank
agent.force_save_memory()

# Clear session memory (for testing)
agent.clear_current_session_memory()

# Export session data
agent.export_current_session("session_backup.json")
```

### 5. Database operations

```python
# Get database statistics
db_stats = agent.get_database_stats()
print(f"Total sessions: {db_stats['total_sessions']}")
print(f"Total messages: {db_stats['total_messages']}")

# Cleanup old data
cleanup_result = agent.cleanup_old_data(days_threshold=30)
print(f"Cleaned {cleanup_result['sessions_cleaned']} old sessions")
```

## Migration từ hệ thống cũ

### Chạy migration tool

```bash
cd /Users/ngocp/Documents/projects/chatbot-rl
python -m src.database.migration_tool
```

Hoặc sử dụng trong code:

```python
from database.migration_tool import DataMigrationTool

migration_tool = DataMigrationTool()

# Run full migration
results = migration_tool.run_full_migration(create_default_session=True)
print(f"Migration results: {results}")

# Verify migration
verification = migration_tool.verify_migration()
print(f"Verification: {verification}")
```

### Dữ liệu được migrate

- ✅ `agent_state.json` → Sessions và messages  
- ✅ `agent_state_meta_learning.pt` → Memory bank entries
- ✅ `experience_buffer.pkl` → Experience sessions
- ✅ Original files được backup tự động

## Web Interface Updates

Streamlit app đã được cập nhật với **Session Management** page:

### Features mới

1. **Session Info**: Hiển thị thông tin session hiện tại
2. **Recent Sessions**: List và resume sessions gần đây  
3. **Database Stats**: Thống kê tổng quát database
4. **Memory Management**: Save/clear memory bank
5. **Advanced Actions**: Meta-learning, cleanup, export

### Sử dụng web interface

```bash
cd /Users/ngocp/Documents/projects/chatbot-rl
streamlit run src/app.py
```

Truy cập trang **"📚 Quản lý Session"** để:
- Xem thông tin session hiện tại
- Chuyển đổi giữa các sessions
- Quản lý memory bank
- Export session data
- Cleanup dữ liệu cũ

## API Reference

### DatabaseManager

```python
from database.database_manager import get_database_manager

db = get_database_manager()

# Session operations
session_id = db.create_session()
session = db.get_session(session_id)
sessions = db.list_sessions(limit=100)

# Message operations  
msg_id = db.add_message(session_id, "user", "Hello")
messages = db.get_chat_history(session_id)
recent = db.get_recent_messages(session_id, count=10)

# Memory bank operations
db.save_memory_bank(session_id, memory_entries, timestep)
entries, timestep = db.load_memory_bank(session_id)
stats = db.get_memory_bank_stats(session_id)

# Maintenance
db.cleanup_old_sessions(days_threshold=30)
stats = db.get_database_stats()
```

### SessionManager

```python
from database.session_manager import get_session_manager

sm = get_session_manager()

# Session lifecycle
session_id = sm.create_new_session(user_id, metadata)
context = sm.get_session_context(session_id)
valid = sm.is_session_valid(session_id)

# Message management
msg_id = sm.add_message_to_session(session_id, role, content, metadata)
history = sm.get_conversation_history(session_id, limit=50)

# Memory operations
sm.save_memory_bank_for_session(session_id, entries, timestep)
entries, timestep = sm.load_memory_bank_for_session(session_id)
```

### MetaLearningEpisodicSystem

```python
from core.meta_learning import MetaLearningEpisodicSystem

# Create với session support
meta_system = MetaLearningEpisodicSystem(
    input_size=768,
    memory_size=1000, 
    session_id=session_id
)

# Store experiences với auto-save
meta_system.store_episodic_experience_with_autosave(
    context="user input",
    response="bot response", 
    reward=0.8
)

# Manual save/load
meta_system.save_memory_to_database(force_save=True)
meta_system.set_session_id(new_session_id)  # Switch session
meta_system.clear_session_memory()  # Reset memory

# Get statistics
stats = meta_system.get_system_statistics()
```

## Testing

Chạy test suite để verify integration:

```bash
cd /Users/ngocp/Documents/projects/chatbot-rl
python test_database_integration.py
```

Test coverage:
- ✅ Database creation và basic operations
- ✅ Session manager functionality  
- ✅ Meta-learning database integration
- ✅ Full agent integration
- ✅ Migration tool verification

## Troubleshooting

### Common Issues

1. **Database not found**
   ```python
   # Database sẽ được tạo tự động tại: data/chatbot_database.db
   # Đảm bảo thư mục data/ tồn tại
   ```

2. **Memory bank not loading**
   ```python
   # Check session_id validity
   sm = get_session_manager()
   if sm.is_session_valid(session_id):
       print("Session valid")
   else:
       print("Invalid session, create new one")
   ```

3. **Migration issues**
   ```python
   # Check existing files
   migration_tool = DataMigrationTool()
   
   # Run verification only
   verification = migration_tool.verify_migration()
   print(verification)
   ```

### Logs và debugging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs sẽ hiển thị:
# - Database operations
# - Session management  
# - Memory save/load operations
# - Migration progress
```

## Performance Notes

- **Auto-save interval**: Memory bank auto-save mỗi 5 experiences
- **Cache TTL**: Session context cache 1 hour
- **Database size**: ~1-2KB per 100 messages
- **Memory bank**: ~50KB per 1000 entries (tensor data)

## Security & Privacy

- **Local SQLite**: Tất cả data được lưu local
- **No external calls**: Database operations không cần internet
- **Backup support**: Original files được backup trước migration
- **Cleanup tools**: Built-in tools để xóa old data

---

🎉 **Hệ thống database integration hoàn tất!** 

Memory bank và chat history giờ đây được lưu trữ persistent và có thể restore chính xác theo session. Chatbot sẽ nhớ context và respond chuẩn xác hơn nhiều!
