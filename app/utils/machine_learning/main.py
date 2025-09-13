import numpy as np

from app.utils.machine_learning.data_preprocessing import (prepare_input,
                                                           splitting_data)
from app.utils.machine_learning.model_index import predicting_result
from app.utils.machine_learning.model_training import MODELS
from app.utils.panic import Panic


def run_option_model(X, Y, x0, model_name):
    """
    Chạy mô hình từ dữ liệu và tên mô hình người dùng chỉ định

    Param:
    - X: Mảng X
    - Y: Mảng Y
    - x0: Mảng dự đoán
    - model_name: Tên mô hình
    """
    X, Y, x0 = prepare_input(X, Y, x0)
    X_train, X_test, Y_train, Y_test = splitting_data(X, Y)

    model = None
    match model_name.lower():
        case "linear" | "linear_regression" | "":
            model = MODELS["linear"]()
        case "lasso" | "lasso_regression":
            model = MODELS["lasso"]()
        case "ridge" | "ridge_regression":
            model = MODELS["ridge"]()
        case "elastic" | "elastic_regression":
            model = MODELS["elastic"]()
        case "polynomial" | "polynomial_regression":
            model = MODELS["linear"]()
        case "bayesian":
            model = MODELS["bayesian"]()
        case "decision_tree":
            model = MODELS["decision_tree"]()
        case "extra_tree":
            model = MODELS["extra_tree"]()
        case "random_forest":
            model = MODELS["random_forest"]()
        case "svm":
            model = MODELS["svr"]()
        case "nu_svm":
            model = MODELS["nu_svr"]()
        case "knn":
            model = MODELS["knn"]()
        case "huber":
            model = MODELS["huber"]()
        case "ransac":
            model = MODELS["ransac"]()
        case "theilsen" | "theil_sen":
            model = MODELS["theilsen"]()
        case _:
            Panic.unreachable("Không có mô hình này")

    model.fit(X_train, Y_train)
    Y_train_predicted = model.predict(X_train)
    Y_test_predicted = model.predict(X_test)

    try:
        y0 = model.predict(x0) if x0 is not None else None
    except Exception as e:
        print(f"Lỗi khi predict với {model_name}: {e}")
        y0 = None

    return predicting_result(
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


def find_best_model(X, Y, x0):
    """
    Tìm mô hình tốt nhất từ tất cả mô hình với dữ liệu X, Y và dự đoán x0

    Param:
    - X: Mảng X
    - Y: Mảng Y
    - x0: Mảng dự đoán

    Return:
    - Dict với thông tin mô hình tốt nhất và kết quả dự đoán
    """
    X, Y, x0 = prepare_input(X, Y, x0)
    X_train, X_test, Y_train, Y_test = splitting_data(X, Y)

    best_model = ""
    best_r2 = -np.inf  # R² càng cao càng tốt

    best_generalization_error = None

    records = run_all_model(X_train, Y_train, X_test, Y_test, x0)

    for record in records:
        r2 = record["r2_test"]
        if r2 > best_r2:
            best_model = record["model"]
            best_r2 = r2
            best_generalization_error = record["generalization_error"]
            best_result = record["y0"]

    if x0 is not None:
        best_result = best_result if best_result else fallback_to_linear(records)

    return {
        "best_model": best_model,
        "best_score": best_r2,
        "best_generalization_error": best_generalization_error,
        "best_result": best_result,
    }
