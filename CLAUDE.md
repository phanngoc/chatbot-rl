# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Main Application
```bash
# Interactive terminal chat
python src/main.py --mode interactive

# Web interface with Streamlit
streamlit run src/app.py

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



The system implements advanced reinforcement learning techniques including Experience Replay, Episodic Memory Consolidation, Elastic Weight Consolidation, and Meta-learning with Memory-Augmented Neural Networks, all integrated with OpenAI's language models for high-quality conversational AI.


## Actions for Debugging
- Create test files under the ./tests directory.
- Log files are stored in the ./logs directory, and each log should include a unique session identifier.
- Before run test file, use command 'source .venv/bin/activate' to activate virtual environment.