import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest


def make_class_models(random_state=0):
    models = {
        "logreg": (
            LogisticRegression(max_iter=5000, penalty="elasticnet", solver="saga"),
            {
                "selector__kbest": [2, 5, 10],
                "model__C": np.logspace(-3, 3, 20),
                "model__l1_ratio": np.linspace(0.1, 1, 10),
            },
        ),
        "svc": (
            SVC(probability=True),
            {
                "selector__kbest": [2, 5, 10],
                "model__C": np.logspace(-3, 3, 20),
                "model__gamma": np.logspace(-5, 1, 20),
                "model__kernel": ["rbf", "sigmoid", "linear"],
            },
        ),
        "rf": (
            RandomForestClassifier(),
            {
                "selector__kbest": [2, 5, 10],
                "model__n_estimators": [100, 300, 600],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", 0.2, 0.5, 0.8],
            },
        ),
        "xgb": (
            XGBClassifier(random_state=1, eval_metric="logloss", tree_method="hist", device="cuda"),
            {
                "selector__kbest": np.arange(5, 50, 1),
                "model__n_estimators": range(1, 1501, 250),
                "model__learning_rate": np.logspace(-4, 0, num=5),
                "model__max_depth": range(1, 21),
                "model__subsample": np.linspace(0.1, 1, num=10),
                "model__colsample_bytree": np.linspace(0.1, 1, num=10),
                "model__colsample_bylevel": np.linspace(0.1, 1, num=10),
                "model__reg_lambda": np.logspace(-3, 3, num=5),
                "model__reg_alpha": np.logspace(-3, 3, num=5),
                "model__gamma": [0, 0.1, 0.2, 0.5],
            },
        ),
    }
    return models


def make_risk_models(random_state=0):
    """
    Risk models for censored survival data.
    Returns dict:
      name -> (estimator, search_space)
    Fit notes:
      - CoxPHSurvivalAnalysis / RandomSurvivalForest expect y = Surv.from_arrays(event, time)
      - XGBRegressor(objective="survival:cox") expects y=time and sample_weight=event
    """

    models = {
        "coxph": (
            CoxPHSurvivalAnalysis(),
            {
                # CoxPHSurvivalAnalysis no tiene muchos hiperparámetros en sksurv
                # Si quieres regularización, usa CoxnetSurvivalAnalysis en su lugar.
            },
        ),
        "rsf": (
            RandomSurvivalForest(
                n_jobs=-1,
                random_state=random_state,
            ),
            {
                "model__n_estimators": [300, 600, 900],
                "model__min_samples_split": [5, 10, 20],
                "model__min_samples_leaf": [5, 15, 30],
                "model__max_features": ["sqrt", 0.3, 0.6, 1.0],
            },
        ),
        "xgb_cox": (
            XGBRegressor(
                objective="survival:cox",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [600, 1200, 2000],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": np.logspace(-3, -1, 10),
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.5, 0.8, 1.0],
                "model__reg_lambda": np.logspace(-3, 2, 10),
                "model__min_child_weight": [1, 5, 10],
            },
        ),
    }

    return models
