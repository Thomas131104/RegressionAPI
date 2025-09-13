import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

from app.utils import run_option_model

# Dữ liệu mẫu
data = fetch_california_housing()
X = data.data
Y = data.target
x0 = np.mean(X, axis=0).reshape(1, -1)


# Danh sách mô hình cần test
MODEL_NAMES = [
    "linear",
    "ridge",
    "lasso",
    "elastic",
    "bayesian",
    "svm",
    "nu_svm",
    "decision_tree",
    "extra_tree",
    "random_forest",
    "knn",
    "huber",
    "ransac",
    "theilsen",
]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_output_keys(model_name):
    result = run_option_model(X, Y, x0, model_name=model_name)

    required_keys = [
        "rmse_train",
        "rmse_test",
        "r2_train",
        "r2_test",
        "x0",
        "y0",
        "generalization_error",
    ]
    for key in required_keys:
        assert key in result, f"Mô hình {model_name} thiếu khóa: {key}"
