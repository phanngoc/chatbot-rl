# Thiết lập môi trường với dotenv

## Cài đặt

Dự án này sử dụng `python-dotenv` để quản lý các biến môi trường. Đảm bảo rằng bạn đã cài đặt tất cả dependencies:

```bash
pip install -r requirements.txt
```

## Thiết lập file .env

1. **Tạo file `.env`** trong thư mục gốc của dự án:

```bash
cp .env.example .env
```

2. **Chỉnh sửa file `.env`** với thông tin của bạn:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.8

# RL Chatbot Configuration
RL_MODEL_NAME=microsoft/DialoGPT-medium
RL_EXPERIENCE_BUFFER_SIZE=5000
RL_MAX_MEMORIES=2000

# Environment
ENVIRONMENT=development
```

## Các biến môi trường

### OpenAI
- `OPENAI_API_KEY`: API key của bạn từ OpenAI (bắt buộc)
- `OPENAI_MODEL`: Model OpenAI để sử dụng (mặc định: gpt-3.5-turbo)
- `OPENAI_TEMPERATURE`: Độ sáng tạo của model (0.0-2.0, mặc định: 0.8)

### RL Chatbot
- `RL_MODEL_NAME`: Tên model HuggingFace (mặc định: microsoft/DialoGPT-medium)
- `RL_EXPERIENCE_BUFFER_SIZE`: Kích thước buffer experience (mặc định: 5000)
- `RL_MAX_MEMORIES`: Số lượng memories tối đa (mặc định: 2000)

### Environment
- `ENVIRONMENT`: Môi trường (development/production, mặc định: development)

## Sử dụng

### Chạy Streamlit app
```bash
cd src
streamlit run app.py
```

### Chạy CLI
```bash
cd src
python main.py --mode interactive
```

### Chạy training
```bash
cd src
python main.py --mode train --data ../examples/sample_training_data.json
```

## Lưu ý bảo mật

- **KHÔNG BAO GIỜ** commit file `.env` vào git
- File `.env` đã được thêm vào `.gitignore`
- Chia sẻ file `.env.example` với team để họ biết cần thiết lập gì
- Sử dụng biến môi trường khác nhau cho development và production

## Troubleshooting

### Lỗi "OPENAI_API_KEY không được thiết lập"
1. Kiểm tra file `.env` có tồn tại không
2. Kiểm tra `OPENAI_API_KEY` có được thiết lập đúng không
3. Đảm bảo file `.env` ở thư mục gốc của dự án

### Lỗi "ModuleNotFoundError: No module named 'dotenv'"
```bash
pip install python-dotenv
```

### Lỗi "openai.error.AuthenticationError"
Kiểm tra API key có hợp lệ không và có đủ credit không
