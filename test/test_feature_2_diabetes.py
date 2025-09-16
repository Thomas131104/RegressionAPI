import requests
import numpy as np
import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# URL API feature 3 (chỉnh theo URL thật)
API_URL = "http://127.0.0.1:8000/regression/best-model/"

# Load dataset Diabetes
diabetes = load_diabetes()
X_data = diabetes.data   # 10 feature
Y_data = diabetes.target # disease progression

# Chia train/test (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=42
)

# Chuyển dữ liệu sang list để JSON serialize
payload = {
    "X_array": X_train.tolist(),
    "Y_array": Y_train.tolist(),
    "x0": X_test.tolist()   # toàn bộ test set
}

# Gửi POST request
response = requests.post(API_URL, json=payload)

# Kiểm tra kết quả
def test_data():
    assert 200 <= response.status_code <= 299
    if response.status_code == 200:
        result = response.json()
        print("Kết quả dự đoán (Diabetes dataset) trên toàn bộ test set:")
        print(json.dumps(result, indent=2))

        assert "rmse_train" in response.json()
        assert "rmse_test" in response.json()
        assert "r2_train" in response.json()
        assert "r2_test" in response.json()

    else:
        print(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
