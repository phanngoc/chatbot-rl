# MANN Tests Directory

This directory contains individual test cases for the Memory-Augmented Neural Network (MANN) system, organized for clear understanding of the training flow and system components.

## Test Structure

### 🏗️ Core Components Tests
- **`test_basic_conversation.py`** - Tests basic chatbot conversation flow
- **`test_memory_search.py`** - Tests embedding-based memory search with fallback
- **`test_memory_management.py`** - Tests memory capacity management and cleanup
- **`test_external_working_memory.py`** - Tests external working memory operations

### 🎯 PPO Training Tests  
- **`test_ppo_training.py`** - Comprehensive PPO training with CSV data loading
- **`test_ppo_importance_ratio.py`** - Tests PPO mathematical components and clipping

### 🔧 System Tests
- **`test_health_monitoring.py`** - Tests health monitoring and performance metrics
- **`test_api_integration.py`** - Tests API client functionality

### 📊 Data Files (`data/`)
- **`ppo_training_data.csv`** - Training questions, answers, and metadata (30 samples)
- **`memory_bank_data.csv`** - Memory initialization data with comprehensive AI/ML content

## Memory Search Functionality

The system now includes **intelligent fallback** for memory search:

### 🔄 OpenAI Embeddings (Primary)
- Uses `text-embedding-3-small` model for semantic similarity
- Requires `OPENAI_API_KEY` environment variable
- Provides high-quality semantic matching

### 🛡️ Fallback System (Backup)  
- **Hash-based vectorization** when OpenAI API is unavailable
- Maintains same interface and functionality
- Uses simple word hashing for similarity calculation
- **Zero breaking** - tests continue to work without API key

## PPO Training Flow

The PPO training test demonstrates the complete training pipeline:

```
1️⃣  Initialize MANN model with external working memory
     ↓
2️⃣  Load memory bank data from CSV (30 diverse AI/ML memories)
     ↓  
3️⃣  Load training questions and reference answers from CSV
     ↓
4️⃣  Forward pass: generate answers with current policy
     ↓
5️⃣  Compute rewards by comparing generated vs reference answers
     ↓
6️⃣  Compute advantages and returns for PPO
     ↓
7️⃣  Update policy using PPO loss (clipped importance ratio)
     ↓
8️⃣  Test improved policy and measure performance gain
```

## Running Tests

### Individual Tests (No API Key Required)
```bash
cd mann_lib/tests

# Core functionality tests
python test_basic_conversation.py
python test_memory_search.py          # Works with/without OPENAI_API_KEY
python test_memory_management.py
python test_external_working_memory.py

# PPO training tests
python test_ppo_training.py           # Enhanced with CSV data
python test_ppo_importance_ratio.py

# System tests  
python test_health_monitoring.py
python test_api_integration.py        # Requires API server running
```

### Complete Test Suite
```bash
cd mann_lib/tests
python run_all_tests.py              # Runs all 8 tests with reporting
```

### With OpenAI API (Enhanced)
```bash
export OPENAI_API_KEY="your-api-key-here"
cd mann_lib/tests
python test_memory_search.py         # Uses OpenAI embeddings
```

## Key Features

### 🔍 Intelligent Memory Search
- **Primary**: OpenAI embeddings for semantic similarity
- **Fallback**: Hash-based vectorization when API unavailable
- **Same interface**: `search_memories(query, top_k, min_similarity)`
- **Graceful degradation**: No breaking changes

### 📊 Comprehensive Data Loading
- CSV-based training data with categories and difficulty levels
- Memory bank initialization with importance weights
- Structured test data for consistent evaluation

### 🐛 Debug Logging
PPO training generates detailed debug logs (`debug_reward_process.log`) containing:
- Input questions, generated answers, reference answers
- Reward computation details and statistics
- Advantage and return calculations
- Training progress information

### 📈 Performance Tracking
- Pre/post-training reward comparison
- Memory utilization statistics
- Training improvement percentages
- Model parameter counts

## Understanding Training Flow

### Memory Operations
- **Write**: `μ̇ᵢ = -zᵢμᵢ + cwzᵢa + zᵢŴqμᵀ`
- **Read**: `Mr = μz, z = softmax(μᵀq)`  
- **Output**: `uad = -Ŵᵀ(σ(V̂ᵀx̃ + b̂v) + Mr) - b̂w`

### PPO Mathematics
- **Importance Ratio**: `r(θ) = π_θ(a|s) / π_θ_old(a|s)`
- **Advantage**: `A = Q(s,a) - V(s)`
- **PPO Loss**: `L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)`

## Data Format

### Training Data CSV
```csv
question,reference_answer,category,difficulty,importance_weight
"Python có ưu điểm gì?","Python là ngôn ngữ lập trình cao cấp...",programming,1,1.0
```

### Memory Bank CSV  
```csv
content,context,tags,importance_weight
"Python là ngôn ngữ lập trình...","programming_context","python,programming",1.2
```

## Troubleshooting

### OpenAI API Issues
- **No API Key**: Tests use fallback system automatically
- **API Errors**: System gracefully degrades to hash-based similarity
- **Rate Limits**: Fallback handles temporary API failures

### Test Failures
- **Check Dependencies**: Ensure `torch`, `numpy`, `openai` installed
- **API Server**: `test_api_integration.py` requires `python run_api.py`
- **Memory Limits**: Some tests use reduced memory sizes for faster execution

## Debug Output

The PPO training test produces:
- 📄 **Debug logs**: Detailed reward process information
- 📊 **Performance metrics**: Training improvement statistics  
- 🧠 **Memory stats**: Utilization and operation counts
- 🎯 **Sample outputs**: Before/after training comparisons

This structure provides **robust, fault-tolerant testing** with **clear insight** into the MANN system's training process and helps debug reward computation issues **without requiring external API dependencies**.