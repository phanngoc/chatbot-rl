# Sơ đồ Experience Buffer Mechanism

## 1. Kiến trúc tổng thể Experience Buffer

```mermaid
graph TB
    subgraph "User Interaction"
        A[User Message] --> B[Context Retrieval]
        B --> C[Response Generation]
        C --> D[User Feedback]
    end
    
    subgraph "Experience Creation"
        D --> E[Create Experience Object]
        E --> F[Calculate Reward]
        F --> G[Set Importance Weight]
    end
    
    subgraph "Experience Buffer Core"
        G --> H[ExperienceReplayBuffer]
        H --> I["Buffer Storage<br/>deque max_size=10000"]
        H --> J["Conversation History<br/>Dict conversation_id List Experience"]
        H --> K["Auto-save<br/>Every 100 experiences"]
    end
    
    subgraph "Sampling & Training"
        I --> L[Weighted Sampling]
        L --> M["Recency Weight<br/>exp minus days_old times 0.1"]
        L --> N["Importance Weight<br/>weight times 1 plus abs reward"]
        M --> O[Batch Selection]
        N --> O
        O --> P[ExperienceReplayTrainer]
        P --> Q[Model Training]
    end
    
    subgraph "Integration Systems"
        H --> R[Memory Manager]
        H --> S[Meta-learning System]
        H --> T[Temporal Weighting]
        R --> U[Intelligent Memory Operations]
        S --> V[Episodic Memory Storage]
        T --> W[Time-based Weighting]
    end
    
    subgraph "Persistence & Cleanup"
        K --> X[Pickle File Storage]
        H --> Y["Auto-cleanup<br/>Remove greater than 30 days old"]
        Y --> Z[Memory Optimization]
    end
```

## 2. Luồng xử lý Experience

```mermaid
sequenceDiagram
    participant U as User
    participant CB as Chatbot
    participant EB as Experience Buffer
    participant MM as Memory Manager
    participant ML as Meta-learning
    participant TW as Temporal Weighting
    
    U->>CB: Send Message
    CB->>CB: Retrieve Context
    CB->>CB: Generate Response
    CB->>U: Return Response
    U->>CB: Provide Feedback (optional)
    
    CB->>EB: Create Experience
    Note over EB: state, action, reward, timestamp, conversation_id
    
    EB->>EB: Add to Buffer
    EB->>EB: Update Conversation History
    EB->>EB: Check Auto-save (every 100)
    
    par Parallel Storage
        EB->>MM: Store via Memory Manager
        EB->>ML: Store in Meta-learning
        EB->>TW: Store in Temporal Weighting
    end
    
    Note over EB: Buffer Statistics Update
    Note over EB: Cleanup Old Experiences (>30 days)
```

## 3. Cơ chế Sampling Strategy

```mermaid
graph TD
    A[Sample Request] --> B{Buffer Size >= 100?}
    B -->|No| C[Return Empty Batch]
    B -->|Yes| D[Calculate Sampling Weights]
    
    D --> E[Recency Calculation]
    E --> F["days_old = current_time minus timestamp"]
    F --> G["recency_weight = exp minus days_old times 0.1"]
    
    D --> H[Importance Calculation]
    H --> I[importance_weight = exp.importance_weight]
    I --> J["reward_factor = 1 plus abs exp.reward"]
    J --> K["importance_weight times equals reward_factor"]
    
    G --> L[Combine Weights]
    K --> L
    L --> M[Normalize Weights]
    M --> N[Weighted Random Sampling]
    N --> O[Return Batch of Experiences]
    
    subgraph "Weight Factors"
        P["Temporal Decay: 0.1"]
        Q["Importance Multiplier: 1.0 to 3.0"]
        R["Reward Amplifier: 1 plus abs reward"]
    end
```

## 4. Tích hợp với Memory Systems

```mermaid
graph LR
    subgraph "Experience Buffer"
        A[ExperienceReplayBuffer]
        B[Buffer Storage]
        C[Conversation History]
    end
    
    subgraph "Memory Manager"
        D[IntelligentMemoryManager]
        E[LLM Extractor]
        F["Memory Operations<br/>ADD UPDATE DELETE NOOP"]
    end
    
    subgraph "Meta-learning"
        G[MetaLearningEpisodicSystem]
        H[MANN Memory]
        I[Episodic Storage]
    end
    
    subgraph "Temporal Weighting"
        J[TemporalWeightingSystem]
        K[Decay Functions]
        L[Weight Updates]
    end
    
    A --> D
    A --> G
    A --> J
    
    D --> M[Knowledge Database]
    G --> N[Memory Bank]
    J --> O[Weighted Experiences]
    
    M --> P[Persistent Knowledge]
    N --> Q[Learned Patterns]
    O --> R[Time-aware Retrieval]
```

## 5. Data Flow trong Experience Buffer

```mermaid
flowchart TD
    subgraph "Input Layer"
        A[User Message]
        B[Bot Response]
        C[User Feedback]
        D[Context Information]
    end
    
    subgraph "Experience Creation"
        E[State: User Message + Context]
        F[Action: Bot Response]
        G[Reward: Feedback Score]
        H[Next State: Updated Context]
        I[Metadata: Timestamp, IDs, etc.]
    end
    
    subgraph "Storage Layer"
        J[Buffer: deque with max_size]
        K[Conversation History: Dict]
        L[Persistent File: pickle]
    end
    
    subgraph "Processing Layer"
        M[Weight Calculation]
        N[Sampling Strategy]
        O[Batch Formation]
    end
    
    subgraph "Output Layer"
        P[Training Batches]
        Q[Statistics]
        R[Memory Integration]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    
    J --> K
    J --> L
    
    J --> M
    M --> N
    N --> O
    
    O --> P
    J --> Q
    J --> R
```

## 6. Performance Metrics và Monitoring

```mermaid
graph TB
    subgraph "Buffer Statistics"
        A[Total Experiences]
        B["Buffer Utilization %"]
        C[Average Reward]
        D["Positive Negative Neutral Count"]
    end
    
    subgraph "Conversation Metrics"
        E[Total Conversations]
        F[Experiences per Conversation]
        G[Conversation Length Distribution]
    end
    
    subgraph "Sampling Metrics"
        H[Sampling Efficiency]
        I[Weight Distribution]
        J[Batch Quality Score]
    end
    
    subgraph "Integration Metrics"
        K[Memory Manager Operations]
        L[Meta-learning Storage]
        M[Temporal Weighting Updates]
    end
    
    A --> N[Performance Dashboard]
    B --> N
    C --> N
    D --> N
    E --> N
    F --> N
    G --> N
    H --> N
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
```

## 7. Lifecycle của Experience

```mermaid
stateDiagram-v2
    [*] --> Created: User Interaction
    Created --> Stored: Add to Buffer
    Stored --> Weighted: Calculate Weights
    Weighted --> Sampled: Training Request
    Sampled --> Used: Model Training
    Used --> Stored: Return to Buffer
    
    Stored --> Archived: Auto-save
    Archived --> Stored: Load from File
    
    Stored --> Cleaned: "Age greater than 30 days"
    Cleaned --> [*]: Removed
    
    Stored --> Updated: Feedback Received
    Updated --> Weighted: Recalculate Weights
    
    note right of Weighted
        Weights based on:
        - Recency temporal decay
        - Importance user feedback
        - Reward magnitude
    end note
```

## 8. Error Handling và Recovery

```mermaid
graph TD
    A[Experience Buffer Operation] --> B{Success?}
    B -->|Yes| C[Continue Normal Flow]
    B -->|No| D[Error Handling]
    
    D --> E{Error Type?}
    E -->|"Storage Full"| F[Trigger Cleanup]
    E -->|"File I/O Error"| G[Retry with Backup]
    E -->|"Memory Error"| H[Reduce Batch Size]
    E -->|"Sampling Error"| I[Fallback to Random]
    
    F --> J[Remove Old Experiences]
    G --> K[Use Alternative Path]
    H --> L[Continue with Smaller Batch]
    I --> M[Use Uniform Sampling]
    
    J --> N[Retry Operation]
    K --> N
    L --> N
    M --> N
    
    N --> O{Success?}
    O -->|Yes| C
    O -->|No| P[Log Error & Continue]
    P --> Q["Graceful Degradation"]
```

Những sơ đồ này minh họa toàn bộ cơ chế hoạt động của Experience Buffer, từ việc tạo experience, lưu trữ, sampling, đến tích hợp với các hệ thống memory khác trong chatbot RL.
