import numpy as np
import pandas as pd

def classify_model(x0, X):
    """
    Chọn model dựa trên vị trí của x0 so với phân vị của X.
    """
    q1 = np.percentile(X, 25, axis=0)
    q2 = np.percentile(X, 50, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1 + 1e-8  # tránh chia 0

    scores = {"stacking": 0, "voting": 0, "elastic": 0}

    for xi, q1_i, q2_i, q3_i, iqr_i in zip(x0, q1, q2, q3, iqr):
        dist = abs(xi - q2_i) / iqr_i  # normalized distance

        if dist <= 0.25:
            scores["stacking"] += 1
        elif dist <= 0.75:
            scores["voting"] += 1
        else:
            scores["elastic"] += 1

    # logic quyết định cuối cùng
    if scores["voting"] > 0 and scores["elastic"] > 0 and scores["stacking"] == 0:
        return "decision_tree"
    else:
        return max(scores, key=scores.get)


# ============================
# Demo
# ============================
np.random.seed(42)
X = np.random.randn(100, 4)  # dataset 100x4
x0_list = [np.random.randn(4) for _ in range(5)]

for i, x0 in enumerate(x0_list):
    model = classify_model(x0, X)
    print(f"Row {i}: data {x0}, model={model}")
