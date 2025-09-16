import requests
import numpy as np
import json

# URL API feature 3 (chỉnh theo URL thật)
API_URL = "http://127.0.0.1:8000/regression/stack-model"

# Dữ liệu test
X_test = np.array([
    [-1.59, -0.60, 0.005, 0.047],
    [-0.45, 0.62, -1.07, -0.14],
    [0.12, 0.51, 0.71, -1.12],
    [-1.53, 1.28, 0.33, -0.74],
    [1.55, 0.12, 1.18, 0.06]
])

Y_test = np.array([
    1.5,
    1.25,
    1.75,
    1.46,
    1.30
])

x0_test = np.array([
    [0.1, -0.2, 0.3, -0.4],
    [-0.5, 0.6, -0.7, 0.8]
])

# Chuyển dữ liệu sang list để JSON serialize
payload = {
    "X_array": X_test.tolist(),
    "Y_array": Y_test.tolist(),
    "x0": x0_test.tolist()
}

def test_data():

    # Gửi POST request
    response = requests.post(API_URL, json=payload)

    # Kiểm tra kết quả
    assert 200 <= response.status_code <= 299
    if response.status_code == 200:
        result = response.json()
        print("Kết quả dự đoán:")
        print(json.dumps(result, indent=2))

        assert "rmse_train" in response.json()
        assert "rmse_test" in response.json()
        assert "r2_train" in response.json()
        assert "r2_test" in response.json()
    
    else:
        print(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
