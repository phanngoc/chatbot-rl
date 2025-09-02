# Episodic Experiences UI Guide

## 🎯 Overview

Streamlit app đã được cập nhật với giao diện hoàn chỉnh để khám phá **episodic experiences** và memory data. Bạn có thể xem chi tiết tất cả experiences, memory bank entries, và session data.

## 🔍 Features mới

### 1. **Session Management Page (📚 Quản lý Session)**

#### Episodic Experiences Section
- **Metrics tổng quan**: Tổng experiences, Memory bank size, Timestep
- **Recent experiences**: Hiển thị 10 experiences gần đây nhất
- **Chi tiết experience**: Context, Response, Reward, User feedback, Timestamp
- **Color coding**: 🟢 Positive reward, 🔴 Negative reward, 🔵 Neutral

#### Memory Bank Information
- **Database status**: Memory bank có được load từ database không
- **Sample memory entries**: Preview top 5 memories với similarity scores
- **Memory details**: Index, Importance weight, Usage count

### 2. **Memory Explorer Page (🔍 Khám phá bộ nhớ)**

Được tái cấu trúc với **4 tabs chuyên biệt**:

#### Tab 1: 🔎 Search Memories
- Tìm kiếm trong retrieval memory system
- Hiển thị similarity, importance, access count
- Tags và metadata details

#### Tab 2: 🧠 Episodic Experiences  
- **Advanced filtering**: Số lượng, sắp xếp theo reward
- **Detailed view**: Context & Response side-by-side
- **Metrics tracking**: Reward, User feedback, Timestamp
- **Quality indicators**: Color-coded reward levels

#### Tab 3: 📚 Memory Bank
- **Search functionality**: Query memory bank trực tiếp
- **Quality assessment**: High/Good/Low quality memories
- **Technical details**: Memory index, tensor shapes, access patterns
- **Similarity analysis**: Top-k relevant memories

#### Tab 4: 📝 Recent Experiences
- **Experience Buffer data**: Direct từ RL training
- **Buffer statistics**: Utilization, Average reward, Conversations
- **Detailed experience view**: State, Action, Reward, Next state
- **Conversation tracking**: Experience IDs và conversation mapping

## 🚀 Cách sử dụng

### 1. Khởi động với sample data

```bash
# Tạo sample data để test
python scripts/test_simple_data.py

# Khởi động Streamlit app
streamlit run src/app.py
```

### 2. Navigate trong UI

1. **Bắt đầu ở "💬 Trò chuyện"** - Chat bình thường
2. **Chuyển sang "📚 Quản lý Session"** - Xem episodic experiences
3. **Explore "🔍 Khám phá bộ nhớ"** - Deep dive vào memory data

### 3. Khám phá Episodic Data

**In Session Management:**
- Xem tổng quan experiences trong current session
- Check memory bank status (database loaded?)
- Review recent episodic experiences với rewards

**In Memory Explorer > Episodic Experiences tab:**
- Filter experiences theo số lượng (5-50)
- Sort by: Mới nhất, Reward cao nhất, Reward thấp nhất
- Analyze reward patterns và user feedback

**In Memory Explorer > Memory Bank tab:**
- Search specific queries trong memory bank
- Understand similarity scores và quality ratings
- Explore memory technical details

## 📊 Data Flow

```
User Chat → RLChatbotAgent → Meta-Learning System
    ↓
Episodic Experience Created
    ↓
Auto-saved to Database (every 5 experiences)
    ↓
Displayed in UI (Session Management & Memory Explorer)
```

## 🎨 UI Elements

### Color Coding
- **🟢 Green**: Positive reward experiences (> 0)
- **🔴 Red**: Negative reward experiences (< 0)  
- **🔵 Blue**: Neutral experiences (= 0)

### Quality Indicators
- **🔥 High Quality**: Similarity > 0.8 + Importance > 1.5
- **⭐ Good Quality**: Similarity > 0.6
- **💡 Low Quality**: Below thresholds

### Metrics
- **Similarity**: Relevance to query (0.0-1.0)
- **Importance Weight**: Learning importance (typically 0.1-2.0)
- **Usage Count**: How often memory được accessed
- **Reward**: RL reward signal (-1.0 to 1.0)

## 🔧 Troubleshooting

### "Chưa có episodic experiences nào"
- Agent chưa process messages nào
- Run sample data script: `python scripts/test_simple_data.py`
- Start chatting để tạo experiences

### "Database không khả dụng"
- Database connection issue
- Run database fix: `python fix_database_constraint.py`
- Check SQLite file exists: `data/chatbot_database.db`

### "Memory bank chưa được load từ database"
- Session chưa có memory data
- Process một vài messages để tạo memories
- Force save: Click "💾 Lưu Memory Bank"

### OpenAI API errors
- Set OPENAI_API_KEY environment variable
- Or create `.env` file với API key
- Some features work without API key (database viewing)

## 📈 Performance Notes

- **Data loading**: Cached để tối ưu performance
- **UI updates**: Real-time refresh khi có new data
- **Memory usage**: Limited display (10-50 items) để tránh lag
- **Database queries**: Optimized với indexes

---

🎉 **Episodic Experiences UI đã sẵn sàng!** 

Bạn có thể explore toàn bộ learning journey của AI agent, từ individual experiences đến memory consolidation patterns!
