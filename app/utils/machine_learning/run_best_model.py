import numpy as np

from app.utils.machine_learning.data_preprocessing import prepare_input, splitting_data
from app.utils.machine_learning.model_index import predicting_result
from app.utils.machine_learning.model_training import MODELS
from app.utils.panic import Panic


def run_all_model(X_train, Y_train, X_test, Y_test, x0=None):
    """
    Chạy tất cả mô hình với dữ liệu đã chia sẵn

    Param:
    - X_train: Mảng X huấn luyện
    - Y_train: Mảng Y huấn luyện
    - X_test: Mảng X kiểm tra
    - Y_test: Mảng Y kiểm tra
    - x0: Mảng dự đoán

    Return:
    - List kết quả dự đoán từ tất cả mô hình
    """

    model_names = ["linear", "elastic", "decision_tree", "random_forest", "svr", "knn"]
    result = []

    for model_name in model_names:
        try:
            model = MODELS[model_name]()
            model.fit(X_train, Y_train)

            Y_train_predicted = model.predict(X_train)
            Y_test_predicted = model.predict(X_test)

            # Dự đoán với x0 nếu có, và xử lý lỗi nếu shape không khớp
            if x0 is not None:
                try:
                    y0 = model.predict(x0)
                except Exception as e:
                    print(f"[{model_name}] predict(x0) lỗi: {e}")
                    y0 = None
            else:
                y0 = None

            result.append(
                predicting_result(
                    model=model_name,
                    X_train=X_train,
                    Y_train=Y_train,
                    Y_train_predicted=Y_train_predicted,
                    X_test=X_test,
                    Y_test=Y_test,
                    Y_test_predicted=Y_test_predicted,
                    x0=x0,
                    y0=y0,
                )
            )

        except Exception as e:
            print(f"[{model_name}] lỗi khi huấn luyện hoặc đánh giá: {e}")
            continue

    return result


def fallback_to_linear(records):
    """
    Nếu mô hình tốt nhất không trả về kết quả dự đoán (y0),
    thì sử dụng kết quả từ mô hình tuyến tính nếu có.

    Param:
    - records: Danh sách kết quả từ tất cả mô hình

    Return:
    - Kết quả dự đoán từ mô hình tuyến tính hoặc None
    """

    for record in records:
        if record["model"] == "linear" and record.get("y0") is not None:
            return record["y0"]
    return None


def run_best_model(X, Y, x0):
    """
    Tìm mô hình tốt nhất từ tất cả mô hình với dữ liệu X, Y và dự đoán x0
    Dựa trên R²_test cao, RMSE_test thấp, generalization_error thấp.
    """
    X, Y, x0 = prepare_input(X, Y, x0)
    X_train, X_test, Y_train, Y_test = splitting_data(X, Y)

    best_model = ""
    best_r2 = -np.inf
    best_rmse = np.inf
    best_generalization_error = np.inf
    best_result = None

    records = run_all_model(X_train, Y_train, X_test, Y_test, x0)

    for record in records:
        r2 = record["r2_test"]
        rmse = record["rmse_test"]
        gen_err = record["generalization_error"]

        # Logic chọn: R² cao nhất, ưu tiên RMSE thấp nếu R² tương đương
        if (r2 > best_r2) or \
           (np.isclose(r2, best_r2, atol=1e-4) and rmse < best_rmse) or \
           (np.isclose(r2, best_r2, atol=1e-4) and np.isclose(rmse, best_rmse, atol=1e-4) and gen_err < best_generalization_error):
            best_model = record["model"]
            best_r2 = r2
            best_rmse = rmse
            best_generalization_error = gen_err
            best_result = record["y0"]

    # Nếu x0 được cung cấp nhưng best_result không có, fallback về linear
    if x0 is not None and best_result is None:
        for record in records:
            if record["model"] == "linear" and record.get("y0") is not None:
                best_result = record["y0"]
                break

    return {
        "best_model": best_model,
        "best_r2_test": best_r2,
        "best_rmse_test": best_rmse,
        "best_generalization_error": best_generalization_error,
        "best_result": best_result,
    }
