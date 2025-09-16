import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils.panic import Panic


# --------------------------------
# Kiểm tra dữ liệu
# --------------------------------
def prepare_input(
    X: list[float] | list[list[float]],
    Y: list[float],
    x0: list[list[float]] | list[float] | None,
):
    """
    Chuẩn bị dữ liệu đầu vào
    Param:
    - X: Mảng X
    - Y: Mảng Y
    - x0: Mảng dự đoán

    Raises:
    - ValueError: Nếu số dòng của X và Y không khớp
    - ValueError: Nếu số cột của x0 không khớp với số cột của X
    - ValueError: Nếu X hoặc Y không phải là mảng 1D hoặc 2D
    - ValueError: Nếu x0 không phải là mảng 1D hoặc 2D (nếu x0 không phải None)

    Return:
    - X: Mảng X đã chuẩn bị
    - Y: Mảng Y đã chuẩn bị
    - x0: Mảng dự đoán đã chuẩn bị (nếu x0 không phải None)
    """

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float).ravel()  # luôn là 1D

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if x0 is not None:
        x0 = np.array(x0, dtype=float)

        if x0.ndim == 1:
            x0 = x0.reshape(1, -1)

        if x0.shape[1] != X.shape[1]:
            raise ValueError(
                f"Số cột của x0 ({x0.shape[1]}) phải bằng số cột của X ({X.shape[1]})"
            )

    return X, Y, x0


# --------------------------------
# Phân chia mô hình
# --------------------------------
def splitting_data(X: np.ndarray, Y: np.ndarray):
    """
    Hàm dùng để phân chia dữ liệu theo tiêu chí:
    - Nếu dữ liệu chỉ có 10 dòng: Training data 100%, Testing data 0%
    - Nếu dữ liệu có từ 10 dòng đến 100 dòng: Training data 95%, Testing data 5%
    - Nếu dữ liệu có từ 100 dòng đến 500 dòng: Training data 90%, Testing data 10%
    - Nếu dữ liệu có từ 500 dòng đến 1000 dòng: Training data 85%, Testing data 15%
    - Nếu dữ liệu có hơn 1000 dòng: Training data 80%, Testing data 20%

    Param:
    - X: Mảng X
    - Y: Mảng Y

    Return:
    - X_train: Mảng X huấn luyện
    - X_test: Mảng X kiểm tra
    - Y_train: Mảng Y huấn luyện
    - Y_test: Mảng Y kiểm tra

    Raises:
    - ValueError: Nếu số dòng của X và Y không khớp
    - Panic: Nếu không thể phân chia dữ liệu (không nên xảy ra)
    """

    # Reshape khi cần
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Phân chia dữ liệu
    if Y.shape[0] < 10:
        return X, X, Y, Y
    elif 10 <= Y.shape[0] <= 100:
        return train_test_split(X, Y, train_size=1 - min(5 / Y.shape[0], 0.2))
    elif 100 <= Y.shape[0] <= 500:
        return train_test_split(X, Y, train_size=0.9)
    elif 500 <= Y.shape[0] <= 1000:
        return train_test_split(X, Y, train_size=0.85)
    elif 1000 <= Y.shape[0]:
        return train_test_split(X, Y, train_size=0.8)
    else:
        Panic.unreachable("Không nên tới nhánh else trong splitting_data")
    