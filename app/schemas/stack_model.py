from typing import Optional, List
from pydantic import BaseModel, field_validator

from app.schemas.utils import Vector, Matrix, VectorOrMatrix
from app.utils.panic import Panic


class InputStackModelData(BaseModel):
    X_array: VectorOrMatrix
    Y_array: Vector
    x0: Optional[VectorOrMatrix]

    # Chuẩn hóa X_array và x0
    # Validator để đảm bảo X_array và x0 luôn là 2D
    @field_validator("X_array", "x0", mode="before")
    @classmethod
    def ensure_2d(cls, v):
        if v is None:
            return None

        if not isinstance(v, list) or not v:
            raise ValueError("X_array/x0 không được rỗng")

        # Nếu là list 1D → nhồi thành ma trận cột
        if all(isinstance(el, (int, float)) for el in v):
            return [[el] for el in v]  # ✅ mỗi phần tử thành một hàng

        # Nếu đã là list of lists
        if all(isinstance(el, list) for el in v):
            for row in v:
                if not all(isinstance(el, (int, float)) for el in row):
                    raise ValueError("Tất cả phần tử trong ma trận phải là số")
            return v

        raise ValueError("Kiểu dữ liệu không hợp lệ cho ma trận")

    @field_validator("Y_array")
    @classmethod
    def check_XY_alignment(cls, y, info):
        X = info.data.get("X_array")
        if X is not None and len(X) != len(y):
            raise ValueError(
                f"Số hàng của X_array ({len(X)}) phải bằng độ dài của Y_array ({len(y)})"
            )
        return y

    # Validator chỉ check số cột của x0 vs X_array
    @field_validator("x0")
    @classmethod
    def check_columns_match(cls, v, info):
        if v is None:
            return None

        X_array = info.data.get("X_array")
        if X_array is not None:
            n_cols_X = len(X_array[0])
            n_cols_x0 = len(v[0])
            if n_cols_X != n_cols_x0:
                raise ValueError(
                    f"Số cột của x0 ({n_cols_x0}) phải bằng số cột của X_array ({n_cols_X})"
                )
        return v


        Panic().todo()


class OutputStackModelData(BaseModel):
    """
    Dữ liệu đầu ra cho tùy chọn mô hình

    Attributes:
    - model: Tên mô hình
    - data_size: Số lượng dòng dữ liệu
    - data_size_label: Nhãn kích thước dữ liệu
    - x0: Ma trận dự đoán (2D, tùy chọn)
    - y0: Vectơ kết quả dự đoán (1D, tùy chọn)
    - rmse_train: Chỉ số RMSE trên tập train
    - rmse_test: Chỉ số RMSE trên tập test
    - mae: Chỉ số MAE trên tập test
    - r2_train: Chỉ số R^2 trên tập train
    - r2_test: Chỉ số R^2 trên tập test
    - r2_status: Trạng thái mô hình
    """

    model: str
    data_size: int
    data_size_label: str
    x0: Optional[VectorOrMatrix] = None
    y0: Optional[Vector] = None
    rmse_train: float
    rmse_test: float
    mae: float
    r2_train: float
    r2_test: float
    r2_status: str
