#!/bin/bash

# Demo script cho RL Chatbot

echo "🚀 RL Chatbot Demo Script"
echo "=========================="

# Tạo thư mục data nếu chưa có
mkdir -p data

# Kiểm tra Python dependencies
echo "📋 Kiểm tra dependencies..."

# Kiểm tra OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    # Kiểm tra file .env
    if [ -f ".env" ]; then
        echo "📁 Tìm thấy file .env, đang load..."
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "⚠️  OPENAI_API_KEY chưa được set và không tìm thấy file .env!"
        echo "   Hãy chạy: python setup_env.py"
        echo "   Hoặc tạo file .env với nội dung: OPENAI_API_KEY=your-api-key-here"
        read -p "Bạn có muốn tiếp tục không? (y/N): " continue_choice
        if [[ $continue_choice != [Yy] ]]; then
            exit 1
        fi
    fi
fi

# Kiểm tra basic dependencies (bỏ transformers vì không cần nữa)
python -c "import torch, streamlit, openai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Một số dependencies chưa được cài đặt."
    echo "   Đang cố gắng cài đặt dependencies cần thiết..."
    
    # Cài đặt dependencies cơ bản trước (bỏ faiss-cpu nếu gây lỗi)
    pip install torch streamlit openai numpy pandas scikit-learn tqdm pydantic python-dotenv
    
    # Thử cài faiss-cpu, nếu lỗi thì dùng alternative
    echo "📦 Đang cài đặt vector search dependencies..."
    pip install faiss-cpu 2>/dev/null || {
        echo "⚠️  faiss-cpu installation failed, sử dụng chromadb thay thế..."
        pip install chromadb sentence-transformers
    }
    
    echo "✅ Dependencies installation completed"
else
    echo "✅ Dependencies OK"
fi

# Menu lựa chọn
echo ""
echo "Chọn chế độ chạy:"
echo "1. Interactive Chat (Terminal)"
echo "2. Web Interface (Streamlit)"  
echo "3. OpenAI API Test (Quick test)"
echo "4. Training với sample data"
echo "5. Evaluation với sample data"
echo "6. Setup Environment (.env)"

read -p "Nhập lựa chọn (1-6): " choice

case $choice in
    1)
        echo "🗣️  Bắt đầu Interactive Chat..."
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        python src/main.py --mode interactive --config configs/default.json
        ;;
    2)
        echo "🌐 Bắt đầu Web Interface..."
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        cd src && streamlit run app.py
        ;;
    3)
        echo "🧪 OpenAI API Test..."
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "❌ OPENAI_API_KEY is required for this test!"
            exit 1
        fi
        
        echo "Testing OpenAI connection..."
        python -c "
from openai import OpenAI
import os

try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Hello! This is a test.'}],
        max_tokens=50
    )
    print('✅ OpenAI API connection successful!')
    print(f'Response: {response.choices[0].message.content}')
    print(f'Tokens used: {response.usage.total_tokens}')
except Exception as e:
    print(f'❌ OpenAI API error: {e}')
    exit(1)
"
        
        echo ""
        echo "🤖 Starting RL Chatbot with OpenAI..."
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from agents.rl_chatbot import RLChatbotAgent
import json

# Load config
with open('configs/default.json', 'r') as f:
    config = json.load(f)

# Initialize agent
agent = RLChatbotAgent(
    openai_model=config.get('openai_model', 'gpt-3.5-turbo'),
    api_key=os.getenv('OPENAI_API_KEY'),
    config=config
)

# Start conversation
agent.start_conversation()
print('🤖 RL Chatbot ready! Type your message:')

# Simple test interaction
result = agent.process_message('Xin chào! Bạn có thể giới thiệu về bản thân không?')
print(f'Bot: {result[\"response\"]}')
print(f'📊 Memories used: {result[\"relevant_memories_count\"]}')
print(f'⏱️ Response time: {result[\"response_time_ms\"]:.2f}ms')
if result.get('openai_usage'):
    usage = result['openai_usage']
    print(f'💰 Tokens: {usage.get(\"total_tokens\", 0)}')
"
        ;;
    4)
        echo "🎓 Bắt đầu Training..."
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        python src/main.py --mode train --data examples/sample_training_data.json --config configs/default.json --save data/trained_agent.json
        ;;
    5)
        echo "🧪 Bắt đầu Evaluation..."
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        python src/main.py --mode eval --data examples/sample_training_data.json --config configs/default.json --load data/trained_agent.json
        ;;
    6)
        echo "🔧 Thiết lập môi trường..."
        python setup_env.py
        ;;
    *)
        echo "❌ Lựa chọn không hợp lệ"
        exit 1
        ;;
esac

echo ""
echo "🎉 Demo completed!"

# Helpful tips
echo ""
echo "💡 Tips:"
echo "   - Để test OpenAI API: python examples/simple_openai_test.py"
echo "   - Để monitor costs: https://platform.openai.com/usage"
echo "   - Nếu gặp lỗi faiss-cpu: system sẽ tự động fallback sang chromadb"
echo "   - Config trong: file .env (sử dụng dotenv)"
echo "   - Thiết lập môi trường: python setup_env.py"
echo "   - Chi tiết dotenv: xem README_ENV.md"
