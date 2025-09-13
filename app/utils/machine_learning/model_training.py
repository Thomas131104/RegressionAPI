import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, HuberRegressor,
                                  LassoCV, LinearRegression, RANSACRegressor,
                                  RidgeCV, TheilSenRegressor)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from app.utils.panic import Panic

# Bảng định nghĩa các tham số của mô hình
PARAMS = {
    "linear": {},

    "ridge": {
        "alphas": np.linspace(0.01, 10, 20)
    },

    "lasso": {
        "alphas": np.logspace(-4, 1, 30),
        "cv": 3,
        "max_iter": 5000
    },

    "elastic": {
        "alphas": np.logspace(-4, 1, 20),
        "l1_ratio": np.linspace(0.1, 1.0, 10),
        "cv": 3,
        "max_iter": 5000
    },

    "bayesian": {},

    "svr": {
        "param_grid": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"]
        }
    },

    "nu_svr": {
        "param_grid": {
            "nu": [0.3, 0.5, 0.7],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"]
        }
    },

    "decision_tree": {
        "param_grid": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },

    "extra_tree": {
        "param_grid": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },

    "random_forest": {
        "param_distributions": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]
        },
        "n_iter": 5
    },

    "knn": {
        "param_grid": {
            "n_neighbors": [1, 3, 5, 7],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    },

    "huber": {
        "max_iter": 1000,
        "epsilon": 1.35
    },

    "ransac": {
        "max_trials": 100,
        "random_state": 42
    },

    "theilsen": {
        "random_state": 42,
        "max_subpopulation": 10000
    }
}



# MODELS: callable khởi tạo mô hình
MODELS = {
    "linear": lambda: LinearRegression(),

    "ridge": lambda: RidgeCV(**PARAMS["ridge"]),

    "lasso": lambda: LassoCV(**PARAMS["lasso"]),

    "elastic": lambda: ElasticNetCV(**PARAMS["elastic"]),

    "bayesian": lambda: BayesianRidge(),

    "svr": lambda: GridSearchCV(
        SVR(),
        PARAMS["svr"]["param_grid"],
        cv=3,
        scoring="r2",
        n_jobs=1
    ),

    "nu_svr": lambda: GridSearchCV(
        NuSVR(),
        PARAMS["nu_svr"]["param_grid"],
        cv=3,
        scoring="r2",
        n_jobs=1
    ),

    "decision_tree": lambda: GridSearchCV(
        DecisionTreeRegressor(),
        PARAMS["decision_tree"]["param_grid"],
        cv=3,
        scoring="r2",
        n_jobs=1
    ),

    "extra_tree": lambda: GridSearchCV(
        ExtraTreeRegressor(),
        PARAMS["extra_tree"]["param_grid"],
        cv=3,
        scoring="r2",
        n_jobs=1
    ),

    "random_forest": lambda: RandomizedSearchCV(
        RandomForestRegressor(),
        param_distributions=PARAMS["random_forest"]["param_distributions"],
        n_iter=PARAMS["random_forest"]["n_iter"],
        cv=3,
        scoring="r2",
        n_jobs=1,
        random_state=42
    ),

    "knn": lambda: GridSearchCV(
        KNeighborsRegressor(),
        PARAMS["knn"]["param_grid"],
        cv=3,
        scoring="r2",
        n_jobs=1
    ),

    "huber": lambda: HuberRegressor(**PARAMS["huber"]),

    "ransac": lambda: RANSACRegressor(**PARAMS["ransac"]),

    "theilsen": lambda: TheilSenRegressor(**PARAMS["theilsen"]),
}
