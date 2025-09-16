#!/bin/bash

echo "🚀 Khởi chạy FastAPI trên Linux..."

# Khởi tạo môi trường ảo nếu chưa có
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy ứng dụng
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
