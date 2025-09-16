import numpy as np
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
from sklearn.tree import DecisionTreeRegressor

from app.utils.machine_learning.classify import classify
from app.utils.machine_learning.data_preprocessing import prepare_input, splitting_data
from app.utils.machine_learning.model_index import predicting_result
from app.utils.panic import Panic


def run_stack_model(X, Y, x0=None):
    X, Y, x0 = prepare_input(X, Y, x0)
    X_train, X_test, Y_train, Y_test = splitting_data(X, Y)
    

    # Xác định số lượng CV dựa vào số lượng records
    n_samples = X.shape[0]

    if n_samples < 5:           # quá ít mẫu, CV không khả thi
        cv_splits = None
    elif n_samples < 50:        # mẫu ít, dùng CV nhỏ
        cv_splits = 2
    else:
        cv_splits = int(np.log2(n_samples))


    # Định nghĩa các mô hình được sử dụng
    decision_tree_model = DecisionTreeRegressor(max_depth=None, random_state=42)

    if n_samples < 100:
        elastic_net_model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42)
    else:
        elastic_net_model = ElasticNetCV(l1_ratio=0.5, random_state=42)
        
    MODELS_IN_THIS_FEATURE = [
        ("elastic", elastic_net_model),
        ("decision_tree", decision_tree_model)
    ]

    


    # Định nghĩa các mô hình Stacking và Voting    
    n_features = X.shape[1]

    if n_features <= 3:
        stacking_model = StackingRegressor(
            estimators=[MODELS_IN_THIS_FEATURE[0]],
            final_estimator=LinearRegression(),
            cv=cv_splits
        )

        voting_model = VotingRegressor(
            estimators=[MODELS_IN_THIS_FEATURE[0]]
        )

    else:
        stacking_model = StackingRegressor(
            estimators=MODELS_IN_THIS_FEATURE,
            final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
            cv=cv_splits
        )

        voting_model = VotingRegressor(
            estimators=MODELS_IN_THIS_FEATURE
        )



    # Fit tất cả các mô hình
    stacking_model.fit(X_train, Y_train)
    voting_model.fit(X_train, Y_train)
    decision_tree_model.fit(X_train, Y_train)
    elastic_net_model.fit(X_train, Y_train)

    # Dự đoán train/test bằng stacking để đánh giá
    Y_train_predicted = stacking_model.predict(X_train)
    Y_test_predicted = stacking_model.predict(X_test)
    
    # Phân loại mỗi x0
    y0 = None
    if x0 is not None:
        x0_classify = classify(X, x0)
        y0 = []

        for i in range(x0.shape[0]):
            x_input = x0[i].reshape(1, -1)  # luôn reshape về (1, n_features)
            label = x0_classify[i]

            match label:
                case "elastic":
                    y0.append(elastic_net_model.predict(x_input)[0])
                case "decision_tree":
                    y0.append(decision_tree_model.predict(x_input)[0])
                case "voting":
                    y0.append(voting_model.predict(x_input)[0])
                case "stacking":
                    y0.append(stacking_model.predict(x_input)[0])
                case _:
                    Panic.unreachable()


    return predicting_result(
        model="Stacking + Voting + Elastic + DecisionTree",
        X_train=X_train,
        Y_train=Y_train,
        Y_train_predicted=Y_train_predicted,
        X_test=X_test,
        Y_test=Y_test,
        Y_test_predicted=Y_test_predicted,
        x0=x0,
        y0=np.array(y0),
    )
