Tôi sẽ phân tích code và tóm tắt cách hệ thống memory hoạt động bằng biểu đồ Mermaid và công thức toán học.

[1 tool called]

Dựa trên phân tích code, tôi sẽ tóm tắt cách memory app hoạt động:

[3 tools called]

## Tóm tắt Memory System - RLChatbot

### 🏗️ **Kiến trúc tổng thể**

Hệ thống memory được thiết kế theo mô hình **Intelligent Memory Manager** thực hiện **Algorithm 1** với các thành phần chính:

1. **RLChatbotAgent** - Agent chính điều phối toàn bộ hệ thống
2. **IntelligentMemoryManager** - Core memory management logic
3. **RetrievalAugmentedMemory** - Vector-based memory storage
4. **Knowledge Database** - Structured knowledge storage
5. **Meta-learning System** - Adaptive learning
6. **Temporal Weighting** - Time-based importance scoring

### 📊 **Công thức toán học chính**

#### **1. Similarity Calculation**
$$
\text{similarity}(q, m) = \frac{q \cdot m}{||q|| \times ||m||}
$$
Trong đó:
- $q$ = query embedding vector
- $m$ = memory embedding vector

#### **2. Memory Operation Decision Logic**
$$
\text{Operation} = \begin{cases}
\text{DELETE} & \text{if } \text{sim} \geq \tau_{\text{delete}} \text{ and } I_{\text{new}} \leq I_{\text{existing}} \\
\text{UPDATE} & \text{if } \tau_{\text{update}} \leq \text{sim} < \tau_{\text{delete}} \text{ and } \Delta I > \theta \\
\text{ADD} & \text{if } \text{sim} < \tau_{\text{update}} \text{ and } \text{capacity} < C_{\text{max}} \\
\text{NOOP} & \text{otherwise}
\end{cases}
$$

Trong đó:
- $\tau_{\text{delete}} = 0.95$ - Delete threshold
- $\tau_{\text{update}} = 0.8$ - Update threshold  
- $I$ - Importance score
- $\theta = 0.3$ - Importance threshold
- $C_{\text{max}} = 5000$ - Max capacity

#### **3. Importance Score Calculation**
$$
I_{\text{total}} = w_1 \cdot I_{\text{content}} + w_2 \cdot I_{\text{context}} + w_3 \cdot I_{\text{temporal}}
$$
Trong đó:
- $I_{\text{content}}$ - Content-based importance từ LLM
- $I_{\text{context}}$ - Context relevance 
- $I_{\text{temporal}}$ - Temporal decay factor

#### **4. Temporal Weighting**
$$
w_{\text{temporal}}(t) = e^{-\lambda \cdot (t_{\text{current}} - t_{\text{memory}})}
$$
Trong đó:
- $\lambda = 0.05$ - Decay rate
- $t$ - Timestamp

#### **5. Memory Capacity Utilization**
$$
U = \frac{N_{\text{current}}}{N_{\text{max}}} \times 100\%
$$
Cleanup trigger khi $U \geq 90\%$

### 🔄 **Quy trình Memory Management**

#### **Phase 1: Information Extraction**
1. **LLM Extraction**: Sử dụng GPT-4o-mini extract key info
   - Entities, intent, key facts, topics
   - Sentiment analysis  
   - Importance scoring (0-1)
   - Memory type classification

#### **Phase 2: Memory Retrieval**  
2. **Vector Search**: ChromaDB retrieval top-k similar memories
3. **Similarity Scoring**: Cosine similarity calculation
4. **Context Analysis**: Evaluate relevance và redundancy

#### **Phase 3: Decision Making (Algorithm 1)**
5. **Operation Decision**: 
   - **ADD**: Thông tin mới, similarity thấp
   - **UPDATE**: Similarity trung bình, có thông tin bổ sung
   - **DELETE**: Similarity cao, thông tin không quan trọng hơn
   - **NOOP**: Đã có memory tương tự đủ tốt

#### **Phase 4: Execution & Storage**
6. **Memory Operations**: Execute decided operation
7. **Knowledge DB**: Store structured knowledge + operation logs
8. **Meta-learning**: Update adaptive parameters
9. **Temporal Weighting**: Apply time-based scoring

### 💾 **Storage Architecture**

```
Memory Storage:
├── ChromaDB (Vector Store)
│   ├── Embeddings (384D/768D)
│   ├── Metadata
│   └── Similarity Search
├── Knowledge Database (SQLite)
│   ├── Extracted Knowledge
│   ├── Memory Operations Log  
│   └── Session Management
└── Experience Buffer (Pickle)
    ├── RL Training Data
    └── Temporal Experiences
```

### 🎯 **Performance Metrics**

**Efficiency Metrics:**
- **ADD Ratio**: $\frac{\text{ADD operations}}{\text{Total operations}}$
- **UPDATE Ratio**: $\frac{\text{UPDATE operations}}{\text{Total operations}}$ 
- **NOOP Ratio**: $\frac{\text{NOOP operations}}{\text{Total operations}}$
- **Memory Utilization**: $\frac{\text{Used capacity}}{\text{Max capacity}} \times 100\%$

**Quality Metrics:**
- **Avg Response Time**: Tracking performance
- **Memory Retrievals**: Usage frequency
- **Consolidation Runs**: Memory optimization events

[1 tool called]

### 🚀 **Key Features**

1. **Intelligent Decision Making**: Algorithm 1 tự động quyết định memory operations
2. **Multi-layered Storage**: Vector + Structured + Temporal storage
3. **LLM-powered Analysis**: Automatic information extraction
4. **Session Management**: Database-persistent conversations  
5. **Adaptive Learning**: Meta-learning với episodic memory
6. **Temporal Decay**: Time-weighted importance scoring
7. **Capacity Management**: Automatic cleanup khi full

Hệ thống này tạo ra một **intelligent memory bank** có khả năng tự quản lý, tối ưu hóa và học hỏi từ interactions, đảm bảo memory quality cao và performance tốt cho conversational AI.