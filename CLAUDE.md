# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Main Application
```bash
# Interactive terminal chat
python src/main.py --mode interactive

# Web interface with Streamlit
streamlit run src/app.py

# Training with data
python src/main.py --mode train --data examples/sample_training_data.json

# Evaluation
python src/main.py --mode eval --data examples/sample_training_data.json
```

### MANN Chatbot (Alternative Interface)
```bash
# Run MANN chatbot CLI
python mann_lib/mann_chatbot.py

# Run MANN API server
python mann_lib/run_api.py

# Simple MANN test
python mann_lib/test_simple.py

# Demo MANN features
python mann_lib/demo.py
```

### Memory Visualization
```bash
# Run memory visualizer demo
python visual_db/demo_memory_visualizer.py

# Run visualizer interface
python visual_db/run_visualizer.py
```

### Testing
- Test scripts are located in the `./tests` directory
- Demo scripts should be created as `demo.py` files for quick feature testing
- Clean up test files after running

## High-Level Architecture

### Core System Components

**Main RL Chatbot (`src/agents/rl_chatbot.py`)**
- Uses OpenAI GPT models (default: gpt-4o-mini) for response generation
- Integrates multiple memory and learning systems
- Supports conversation management and user feedback (1-5 star rating system)

**MANN Alternative (`mann_lib/`)**
- Standalone Memory-Augmented Neural Network implementation
- Provides CLI chatbot interface (`mann_chatbot.py`)
- API server (`run_api.py`) and monitoring capabilities
- External working memory management with episodic storage

### Memory Systems

**Experience Replay (`src/core/experience_replay.py`)**
- Stores (state, action, reward, next_state) tuples
- Prioritized sampling based on importance and recency
- Automatic buffer persistence

**Retrieval-Augmented Memory (`src/memory/retrieval_memory.py`)**
- Vector search using FAISS or ChromaDB
- Embedding-based similarity search with temporal decay
- Forgetting mechanism for memory management

**Memory Consolidation (`src/memory/consolidation.py`)**
- LLM-powered summarization of experiences using OpenAI API
- Knowledge graph integration for concept relationships
- Episodic to semantic memory conversion

**Meta-learning (`src/core/meta_learning.py`)**
- Memory-Augmented Neural Network patterns
- Episodic memory with attention mechanisms
- Learning to select relevant memories

**Temporal Weighting (`src/core/temporal_weighting.py`)**
- Multiple decay functions (exponential, power-law, forgetting curve)
- Importance weighting based on feedback and usage patterns
- Access pattern analysis

### Database Integration

**Database Manager (`src/database/database_manager.py`)**
- SQLite-based persistence for conversations and experiences
- Session management and migration tools
- Knowledge database with semantic search capabilities

### Key Configuration Rules

From `.cursor/rules/`:
- Do not use `hasattr()` for attribute checking
- Avoid fallback mechanisms
- Use LaTeX syntax for math blocks in markdown
- Use `< >` instead of `( )` for Mermaid diagrams
- Documentation and installation guides are in `./docs` directory

## Environment Setup

Required environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Optional configuration via environment variables:
- `RL_MODEL_NAME` (default: "microsoft/DialoGPT-medium")
- `RL_EXPERIENCE_BUFFER_SIZE` (default: "10000")  
- `RL_MAX_MEMORIES` (default: "5000")
- `OPENAI_TEMPERATURE` (default: "0.8")

## Project Structure

- `src/` - Main application code
  - `agents/` - RL chatbot implementation
  - `core/` - Core algorithms (experience replay, meta-learning, etc.)
  - `memory/` - Memory management systems
  - `database/` - Database integration and managers
- `mann_lib/` - Standalone MANN implementation
- `tests/` - Test scripts and demos
- `visual_db/` - Memory visualization tools
- `docs/` - Comprehensive documentation
- `scripts/` - Utility scripts

The system implements advanced reinforcement learning techniques including Experience Replay, Episodic Memory Consolidation, Elastic Weight Consolidation, and Meta-learning with Memory-Augmented Neural Networks, all integrated with OpenAI's language models for high-quality conversational AI.