from typing import Optional

from pydantic import BaseModel, field_validator

from app.schemas.utils import Matrix, Vector, VectorOrMatrix
from app.utils.panic import Panic


class InputBestModel(BaseModel):
    """
    Schema cho input của best_model

    Param:
    - X_array: Mảng X (bắt buộc)
    - Y_array: Mảng Y (bắt buộc)
    - x0: Mảng dự đoán (tùy chọn)

    Note:
    - X_array và x0 luôn là ma trận 2D
    - Y_array luôn là vector 1D
    - Số cột của x0 phải bằng số cột của X_array
    - Số hàng của Y_array phải bằng số hàng của X_array

    Raises:
    - ValueError: Nếu dữ liệu không hợp lệ
    - Panic: Nếu có lỗi không lường trước
    """

    X_array: VectorOrMatrix
    Y_array: Vector
    x0: Optional[VectorOrMatrix] = None

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


class OutputBestModel(BaseModel):
    """
    Schema cho output của best_model

    Param:
    - best_model: Tên mô hình tốt nhất
    - best_score: Chỉ số RMSE trên tập test của mô hình tốt nhất
    - best_generalization_error: Sai số tổng quát hóa của mô hình tốt nhất
    - best_result: Kết quả dự đoán từ mô hình tốt nhất (nếu x0 được cung cấp)
    """

    best_model: str
    best_score: float
    best_generalization_error: float
    best_result: Optional[Vector] = None
