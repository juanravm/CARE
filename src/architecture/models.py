import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from xgboost.core import XGBoostError
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def make_class_models(random_state=0):
    models = {
        "logreg": (
            LogisticRegression(max_iter=5000, penalty="elasticnet", solver="saga"),
            {
                "selector__kbest": [5, 10, 20],
                "model__C": np.logspace(-2, 2, 8),
                "model__l1_ratio": np.linspace(0.1, 1, 5),
            },
        ),
        "svc": (
            SVC(probability=True),
            {
                "selector__kbest": [10, 20, 40],
                "model__C": np.logspace(-2, 2, 6),
                "model__gamma": np.logspace(-4, -1, 4),
                "model__kernel": ["rbf", "linear"],
            },
        ),
        "rf": (
            RandomForestClassifier(),
            {
                "selector__kbest": [10, 20, 40],
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "xgb": (
            XGBClassifier(
                random_state=1,
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
            ),
            {
                "selector__kbest": [10, 20, 40],
                "model__n_estimators": [300, 600, 900, 1200],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.25],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.5, 0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
                "model__reg_lambda": np.logspace(-3, 3, num=5),
                "model__reg_alpha": np.logspace(-3, 3, num=5),
                "model__gamma": [0, 0.1, 0.2],
            },
        ),
    }
    return models


class CoxXGBRegressor(XGBRegressor):
    """
    Wrapper de XGBRegressor para aceptar y con campos event/time (Surv o DataFrame)
    y mapear event -> sample_weight y time -> target.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        event = None
        time = y
        X_in = X

        # Surv o array estructurado
        if hasattr(y, "dtype") and getattr(y.dtype, "names", None):
            names = set(y.dtype.names)
            if {"event", "time"}.issubset(names):
                event = np.asarray(y["event"], dtype=float)
                time = np.asarray(y["time"], dtype=float)
        # DataFrame con columnas event/time
        elif isinstance(y, pd.DataFrame) and {"event", "time"}.issubset(set(y.columns)):
            event = y["event"].to_numpy(dtype=float)
            time = y["time"].to_numpy(dtype=float)

        if event is not None and sample_weight is None:
            sample_weight = event

        try:
            return super().fit(X_in, time, sample_weight=sample_weight, **kwargs)
        except XGBoostError as e:
            # Fallback a CPU si no hay soporte GPU o si el input es CuPy y XGBoost no lo acepta
            if "GPU" in str(e) or "gpu" in str(e) or "device" in str(e):
                X_cpu = cp.asnumpy(X_in) if cp is not None and hasattr(X_in, "__cuda_array_interface__") else X_in
                cpu_params = self.get_params()
                cpu_params.update({"tree_method": "hist"})
                cpu_params.pop("predictor", None)
                cpu_params.pop("device", None)
                self.set_params(**cpu_params)
                return super().fit(X_cpu, time, sample_weight=sample_weight, **kwargs)
            raise


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
                "selector__kbest": [5, 10, 15],
                "model__alpha": np.logspace(-4, 2, 10),
            },
        ),
        "rsf": (
            RandomSurvivalForest(
                n_jobs=-1,
                random_state=random_state,
            ),
            {
                "selector__kbest": [10, 20, 40],
                "model__n_estimators": [300, 600],
                "model__min_samples_split": [5, 10],
                "model__min_samples_leaf": [5, 15],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "xgb_cox": (
            CoxXGBRegressor(
                objective="survival:cox",
                tree_method="hist",
                random_state=random_state,
                device="cuda",
            ),
            {
                "selector__kbest": [10, 20, 40],
                "model__n_estimators": [400, 800],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_lambda": [1, 3, 10],
                "model__min_child_weight": [1, 5],
            },
        ),
    }

    return models
