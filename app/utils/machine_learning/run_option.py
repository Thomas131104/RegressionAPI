from app.utils.machine_learning.data_preprocessing import prepare_input, splitting_data
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
