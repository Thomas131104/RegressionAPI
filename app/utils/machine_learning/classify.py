import numpy as np
import pandas as pd

def classify(X: np.ndarray, x0: np.ndarray):
    """
    Phân loại mỗi dòng trong x0 theo meta-logic (stacking, voting, elastic, decision_tree).
    
    Parameters
    ----------
    X : np.ndarray
        Dữ liệu gốc (n_samples, n_features)
    x0 : np.ndarray
        Dữ liệu mới cần phân loại (m_samples, n_features)
    
    Returns
    -------
    labels : list[str]
        Nhãn (tên mô hình) cho mỗi dòng trong x0
    """

    # Tính Q1, Q2, Q3 và std cho từng feature
    Q1 = np.percentile(X, 25, axis=0)
    Q2 = np.percentile(X, 50, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    std = X.std(axis=0)

    # Ngưỡng "gần"
    thresh = 0.5 * std

    labels = []
    for row in x0:
        near_Q2, near_Q13, far = 0, 0, 0

        for j, val in enumerate(row):
            if abs(val - Q2[j]) <= thresh[j]:
                near_Q2 += 1
            elif abs(val - Q1[j]) <= thresh[j] or abs(val - Q3[j]) <= thresh[j]:
                near_Q13 += 1
            else:
                far += 1

        n_feat = row.shape[0]

        # Logic phân loại
        if n_feat == 1:
            if near_Q2 == 1:
                labels.append("stacking")
            elif near_Q13 == 1:
                labels.append("voting")
            else:
                labels.append("elastic")

        elif n_feat == 2:
            if near_Q2 == 2:
                labels.append("stacking")
            elif near_Q2 == 1 and near_Q13 == 1:
                labels.append("stacking")
            elif near_Q13 == 2:
                labels.append("voting")
            elif near_Q2 == 1 and far == 1:
                labels.append("voting")
            elif near_Q13 == 1 and far == 1:
                labels.append("elastic")
            else:  # far == 2
                labels.append("elastic")

        else:  # n_feat >= 3
            if near_Q2 >= (2 * n_feat // 3):
                labels.append("stacking")
            elif near_Q2 >= (n_feat // 3) and near_Q13 >= (n_feat // 3):
                labels.append("stacking")
            elif near_Q13 >= (2 * n_feat // 3):
                labels.append("voting")
            elif near_Q2 >= (n_feat // 3) and far >= (n_feat // 3):
                labels.append("voting")
            elif near_Q13 >= (n_feat // 3) and far >= (n_feat // 3):
                labels.append("decision_tree")
            elif far >= (2 * n_feat // 3):
                labels.append("elastic")
            else:
                if near_Q2 >= near_Q13 and near_Q2 >= far:
                    labels.append("stacking")
                elif near_Q13 >= near_Q2 and near_Q13 >= far:
                    labels.append("voting")
                else:
                    labels.append("elastic")
    return labels
