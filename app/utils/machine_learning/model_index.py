import numpy as np
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)


def status_of_model(
    r2_train,
    r2_test,
    rmse_train,
    rmse_test,
    r2_threshold=0.5,
    rmse_gap_threshold=0.2,
    r2_gap_threshold=0.2,
):
    """
    Đánh giá tình trạng mô hình dựa trên cả R² và RMSE của tập huấn luyện và kiểm tra.

    - Overfit: R² train cao hơn test nhiều, RMSE test cao hơn train nhiều
    - Underfit: R² thấp ở cả hai tập, RMSE cao ở cả hai tập
    - Normal: Không có dấu hiệu rõ ràng của overfit hay underfit
    """
    r2_gap = abs(r2_train - r2_test)
    rmse_gap = abs(rmse_test - rmse_train)

    # Kiểm tra độ chính xác tuyệt đối
    if r2_train < r2_threshold and r2_test < r2_threshold:
        return "Weak model"

    # Kiểm tra độ chênh lệch
    if r2_gap > r2_gap_threshold and rmse_gap > rmse_gap_threshold:
        return "Overfit"
    elif r2_gap < r2_gap_threshold and rmse_gap < rmse_gap_threshold:
        return "Stable"

    return "Uncertain"


def composite_score(rmse_train, rmse_test, r2_train, r2_test):
    """
    Tính chỉ số tổng hợp để đánh giá mức độ generalization của model theo cách: chuẩn hóa trước khi cộng.

    Param:
    - rmse_train: Chỉ số RMSE trên tập train
    - rmse_test: Chỉ số RMSE trên tập test
    - r2_train: Chỉ số R² trên tập train
    - r2_test: Chỉ số R² trên tập test

    Return:
    - Chỉ số tổng hợp (composite score)
    0 <= composite score <= 1 (càng thấp càng tốt)

    Cách tính:
    - Sai số RMSE giữa train và test (chuẩn hóa)
    - Sai số R² giữa train và test
    - Trung bình cộng của hai sai số trên
    """

    # chuẩn hóa RMSE (chia cho max của train/test để đưa về [0,1])
    sum_rmse = rmse_train + rmse_test
    diff_rmse = (
        abs(rmse_train / sum_rmse - rmse_test / sum_rmse) if sum_rmse >= 1e-4 else 0
    )
    diff_r2 = abs(r2_train - r2_test)
    return (diff_rmse + diff_r2) / 2


# Hàm làm tròn số
def to_float_round(val, precision=4):
    """
    Hàm để làm tròn số

    Param:
    - val: Giá trị
    - precision: Độ chính xác

    Return:
    - Giá trị đã làm tròn
    """

    if isinstance(val, (np.ndarray, list)):
        arr = np.array(val, dtype=np.float32)
        return (
            np.round(arr, precision).astype(float).tolist()
        )  # important: float Python
    else:
        return round(float(val), precision)


def get_data_size_label(n: int) -> str:
    """
    Trả về kích thước dữ liệu
    """
    if n < 50:
        return "tiny"
    elif n < 100:
        return "small"
    elif n < 1000:
        return "normal"
    elif n < 10000:
        return "big"
    else:
        return "enormous"


def predicting_result(
    model: str,
    X_train,
    Y_train,
    Y_train_predicted,
    X_test,
    Y_test,
    Y_test_predicted,
    x0=None,
    y0=None,
):
    """
    Dictionary kết quả các mô hình đang học

    Parameters:
    - model: Tên mô hình
    - X_train, Y_train, Y_train_predicted, X_test, Y_test, Y_test_predicted: Các mảng X và Y
    - x0, y0: Các giá trị dự đoán

    Returns:
    - model: Tên mô hình,
    - data_size: Số lượng dòng dữ liệu
    - x0: Mảng cần dự đoán kết quả
    - y0: Mảng chứa kết quả dự đoán từ mô hình
    - rmse_train: Chỉ số RMSE trên tập train
    - rmse_test: Chỉ số RMSE trên tập test
    - mae: Chỉ số MAE trên tập test
    - r2_train: Chỉ số R^2 trên tập train
    - r2_test: Chỉ số R^2 trên tập test
    - delta: Sai số từ các chỉ số RMSE và R^2
    - status: Trạng thái mô hình
    """

    data_size = X_train.shape[0] + X_test.shape[0]
    rmse_train = root_mean_squared_error(Y_train, Y_train_predicted)
    rmse_test = root_mean_squared_error(Y_test, Y_test_predicted)
    mae = mean_absolute_error(Y_test, Y_test_predicted)
    r2_train = r2_score(Y_train, Y_train_predicted)
    r2_test = r2_score(Y_test, Y_test_predicted)

    result = {
        "model": model,
        "data_size": data_size,
        "data_size_label": get_data_size_label(data_size),
        "rmse_train": to_float_round(rmse_train),
        "rmse_test": to_float_round(rmse_test),
        "mae": to_float_round(mae),
        "r2_train": to_float_round(r2_train),
        "r2_test": to_float_round(r2_test),
        "r2_status": status_of_model(r2_train, r2_test, rmse_train, rmse_test),
        "generalization_error": to_float_round(
            composite_score(rmse_train, rmse_test, r2_train, r2_test)
        ),
        "generalization_status": status_of_model(
            r2_train, r2_test, rmse_train, rmse_test
        ),
    }

    if x0 is not None and y0 is not None:
        result["x0"] = x0.tolist() if hasattr(x0, "tolist") else x0
        result["y0"] = to_float_round(y0.tolist()) if hasattr(y0, "tolist") else y0

    return result
