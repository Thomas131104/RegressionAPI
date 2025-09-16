Write-Host "🚀 Khởi chạy FastAPI trên Windows..."

# Khởi tạo môi trường ảo
if (!(Test-Path ".venv")) {
    python -m venv .venv
}

# Kích hoạt môi trường ảo nếu có
.\.venv\Scripts\Activate.ps1

# Tải các thư viện cần thiết
pip install -r .\requirements.txt

# Chạy app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
