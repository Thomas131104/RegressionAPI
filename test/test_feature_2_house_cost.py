import requests
import numpy as np
import json
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# URL API feature 3 (chỉnh theo URL thật)
API_URL = "http://127.0.0.1:8000/regression/best-model/"

# Load dataset California housing
california = fetch_california_housing()
X_data = california.data       # 8 feature
Y_data = california.target     # median house value

# Chia train/test (80% train, 20% test)
X, x0, Y, y0 = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=42
)

def test_data():
    # Chuyển dữ liệu sang list để JSON serialize
    payload = {
        "X_array": X.tolist(),
        "Y_array": Y.tolist(),
        "x0": x0.tolist()   # toàn bộ test set
    }

    # Gửi POST request
    response = requests.post(API_URL, json=payload)

    # Kiểm tra kết quả
    assert 200 <= response.status_code <= 299
    if response.status_code == 200:
        result = response.json()
        print("Kết quả dự đoán giá nhà (California Housing) trên toàn bộ test set:")
        print(json.dumps(result, indent=2))

        assert "rmse_train" in response.json()
        assert "rmse_test" in response.json()
        assert "r2_train" in response.json()
        assert "r2_test" in response.json()
    else:
        print(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
