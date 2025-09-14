# Regression API

## Giới thiệu

Regression API là một RESTful API sử dụng FastAPI cho phép người dùng thực hiện các bài toán hồi quy với nhiều mô hình học máy khác nhau (Linear Regression, Ridge, Lasso, ElasticNet, SVM, Decision Tree, Random Forest, KNN, v.v.). API hỗ trợ đánh giá mô hình, dự đoán giá trị mới, và lưu trữ lịch sử kết quả vào MongoDB.

---

## Tính năng

- Chạy mô hình hồi quy với dữ liệu đầu vào tuỳ chọn
- Dự đoán giá trị mới với các mô hình khác nhau
- Đánh giá chất lượng mô hình (RMSE, MAE, R², trạng thái overfit/underfit)
- Tìm mô hình tốt nhất cho bộ dữ liệu
- Lưu lịch sử kết quả vào MongoDB, tự động xoá bản ghi cũ khi vượt quá giới hạn
- Quản lý vòng đời kết nối database tự động
- Kiểm thử tự động với pytest

---

## Cấu trúc thư mục

```
MachineLearningAPI/
├── app/
│   ├── app.py # Khởi tạo FastAPI app
│   ├── __init__.py
│   ├── database/ # Kết nối và truy vấn MongoDB
│   ├── router/ # Định nghĩa các route API
│   ├── schemas/ # Định nghĩa schema đầu vào/ra (Pydantic)
│   └── utils/ # Xử lý logic machine learning, tiện ích
├── test/ # Unit test cho các tính năng
├── main.py # Chạy server bằng Hypercorn
├── requirements.txt # Thư viện phụ thuộc
├── .env # Biến môi trường (URL MongoDB)
├── README.md # Tài liệu hướng dẫn
├── pytest.ini # Cấu hình pytest
└── .vscode/ # Cấu hình VSCode
```

---

## Hướng dẫn cài đặt

1. **Clone repository**

   ```sh
   git clone <repo-url>
   cd MachineLearningAPI
   ```

2. **Tạo môi trường ảo và cài đặt dependencies**

   ```sh
   python -m venv venv
   source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows
   pip install -r requirements.txt
   ```

3. **Cấu hình MongoDB**

   - Tạo file `.env` với biến `URL` chứa connection string MongoDB (xem ví dụ trong repo).

4. **Chạy server**
   - Sử dụng Hypercorn:
     ```sh
     python main.py
     ```
   - Hoặc dùng VSCode task `"Run FastAPI"` để chạy bằng Uvicorn (xem ).

## Sử dụng API

### 1. Tùy chọn mô hình hồi quy

- **Endpoint:** `POST /regression/option/`
- **Input (JSON):**
  ```json
  {
    "X_array": [[50, 1], [60, 2], ...],
    "Y_array": [150000, 180000, ...],
    "x0": [[85, 3]],
    "model": "linear"
  }
  ```

---

- **Output:**
  ```json
  {
    "model": "linear",
    "data_size": 6,
    "data_size_label": "tiny",
    "x0": [[85, 3]],
    "y0": [giá trị dự đoán],
    "rmse_train": ...,
    "rmse_test": ...,
    "mae": ...,
    "r2_train": ...,
    "r2_test": ...,
    "r2_status": "Stable"
  }
  ```

### 2. Tìm mô hình tốt nhất

- **Endpoint:** `POST /regression/best-model-without-prediction/`
- **Input (JSON):**
  ```json
  {
    "X_array": [[...], ...],
    "Y_array": [...],
    "x0": [[...]]
  }
  ```
- **Output:**
  ```json
  {
    "best_model": "random_forest",
    "best_score": ...,
    "best_generalization_error": ...,
    "best_result": [giá trị dự đoán]
  }
  ```

### 3. Lịch sử kết quả

- **Endpoint:** `GET /regression/option/history` hoặc `/regression/best-model/history`
- **Trả về:** Danh sách các bản ghi lịch sử (giới hạn 100 bản ghi mới nhất).

---

## Kiểm thử

- Chạy toàn bộ test với pytest:
  ```sh
  pytest
  ```
- Các test nằm trong thư mục kiểm tra các endpoint và logic mô hình.

---

## Đóng góp

- Fork và tạo pull request nếu bạn muốn đóng góp thêm mô hình hoặc cải tiến API.

---

**Liên hệ:**

- # Tác giả: Mus
