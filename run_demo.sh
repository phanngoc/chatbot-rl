#!/bin/bash

# Demo script cho RL Chatbot

echo "🚀 RL Chatbot Demo Script"
echo "=========================="

# Tạo thư mục data nếu chưa có
mkdir -p data

# Kiểm tra Python dependencies
echo "📋 Kiểm tra dependencies..."
python -c "import torch, transformers, streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Một số dependencies chưa được cài đặt. Chạy: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies OK"

# Menu lựa chọn
echo ""
echo "Chọn chế độ chạy:"
echo "1. Interactive Chat (Terminal)"
echo "2. Web Interface (Streamlit)"
echo "3. Training với sample data"
echo "4. Evaluation với sample data"

read -p "Nhập lựa chọn (1-4): " choice

case $choice in
    1)
        echo "🗣️  Bắt đầu Interactive Chat..."
        cd src && python main.py --mode interactive --config ../configs/default.json
        ;;
    2)
        echo "🌐 Bắt đầu Web Interface..."
        cd src && streamlit run app.py
        ;;
    3)
        echo "🎓 Bắt đầu Training..."
        cd src && python main.py --mode train --data ../examples/sample_training_data.json --config ../configs/default.json --save ../data/trained_agent.json
        ;;
    4)
        echo "🧪 Bắt đầu Evaluation..."
        cd src && python main.py --mode eval --data ../examples/sample_training_data.json --config ../configs/default.json --load ../data/trained_agent.json
        ;;
    *)
        echo "❌ Lựa chọn không hợp lệ"
        exit 1
        ;;
esac
