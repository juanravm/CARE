import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier


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
    }

    return models


class StackingModel:
    def __init__(self, logres_model, svc_model, rf_model, meta_model=None, cv=5):
        """
        Initialize the stacking models with the base models
        """
        self.logres_model = logres_model
        self.svc_model = svc_model
        self.rf_model = rf_model
        self.meta_model = meta_model if meta_model else LogisticRegression(random_state=1, max_iter=1000)
        self.cv = cv
        self.stack_clf = None

    def fit(self, X, y):
        """
        Train stacking model with data X, y
        """
        base_learners = [
            ("logres", clone(self.logres_model)),
            ("svc", clone(self.svc_model)),
            ("rf", clone(self.rf_model)),
        ]
        self.stack_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=self.meta_model,
            cv=self.cv,
            n_jobs=-1,
            passthrough=False,
        )
        self.stack_clf.fit(X, y)

    def predict(self, X):
        """
        Predict with the trained model
        """
        if self.stack_clf is None:
            raise ValueError("Model must be trained with .fit(X, y) before predicting.")
        return self.stack_clf.predict(X)

    def predict_proba(self, X):
        """
        Return the prediction probabilities
        """
        if self.stack_clf is None:
            raise ValueError("Model must be trained with .fit(X, y) before predicting.")
        return self.stack_clf.predict_proba(X)
