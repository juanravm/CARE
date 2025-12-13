import json
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from src.architecture.models import make_class_models, make_risk_models
from src.preproc.preproc_utils import FCBFSelector
from src.utils.utils import setup_hyperparameters

# 0) Setup run hyperparameters and W&B run
config_path = "src/configs/config.yaml"
_, run = setup_hyperparameters(config_path)

# 1) Loading data
df = pd.read_csv("data/training_df.csv")

# Filtering variables with too many missing values
df = df.loc[:, df.isna().mean() <= 0.1]

# Removing samples with any missing value
df = df.dropna()
print("Training with {} samples".format(df.shape[0]))

y_class = df["risk_status"].copy()
y_event = df["dfs_status"].copy()
y_time = df["dfs_time"].copy()
X = df.drop(columns=["risk_status", "dfs_status", "dfs_time"])
y = y_class
y_surv = Surv.from_arrays(event=y_event.astype(bool), time=y_time)

# 4) Selector for best features
selector = FCBFSelector(mode="rank", threshold=0.0, n_bins=2)

# 5) Models and hyperparameters search
class_models = make_class_models()
risk_models = make_risk_models()

cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorer_class = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")
cv_risk = KFold(n_splits=5, shuffle=True, random_state=0)


def _cindex_scorer(y_true, y_pred):
    """
    Concordance index scorer for survival models.
    Assumes y_true is a structured array with fields ('event', 'time').
    Uses negative scores so that higher hazard -> lower survival time.
    """
    return concordance_index_censored(y_true["event"], y_true["time"], -y_pred)[0]


scorer_risk = make_scorer(_cindex_scorer, greater_is_better=True, needs_proba=False)

# 6) Class models
class_results = {}
class_table_rows = []
for name, (estimator, search_space) in class_models.items():
    use_selector = any(p.startswith("selector__") for p in search_space)
    steps = [("model", estimator)]
    if use_selector:
        steps.insert(0, ("selector", selector))
    pipe = Pipeline(steps)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=search_space,
        n_iter=20,
        cv=cv_class,
        scoring=scorer_class,
        n_jobs=-1,
        random_state=0,
        refit=True,
    )
    search.fit(X, y)
    class_results[name] = {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }
    print(f"{name}: AUC={search.best_score_:.3f}")
    if run:
        run.log(
            {
                f"class/{name}/best_auc": search.best_score_,
                f"class/{name}/best_params": json.dumps(search.best_params_, default=str),
            }
        )
        class_table_rows.append([name, search.best_score_, json.dumps(search.best_params_, default=str)])

# 7) Risk models
risk_results = {}
risk_table_rows = []
for name, (estimator, search_space) in risk_models.items():
    use_selector = any(p.startswith("selector__") for p in search_space)
    steps = [("model", estimator)]
    if use_selector:
        steps.insert(0, ("selector", selector))
    pipe = Pipeline(steps)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=search_space,
        n_iter=50,
        cv=cv_risk,
        scoring=scorer_risk,
        n_jobs=-1,
        random_state=0,
        refit=True,
    )
    search.fit(X, y_surv)
    risk_results[name] = {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }
    print(f"{name}: C-index={search.best_score_:.3f}")
    if run:
        run.log(
            {
                f"risk/{name}/best_cindex": search.best_score_,
                f"risk/{name}/best_params": json.dumps(search.best_params_, default=str),
            }
        )
        risk_table_rows.append([name, search.best_score_, json.dumps(search.best_params_, default=str)])

if run:
    if class_table_rows:
        run.log({"class_results": wandb.Table(columns=["model", "best_auc", "best_params"], data=class_table_rows)})
    if risk_table_rows:
        run.log({"risk_results": wandb.Table(columns=["model", "best_cindex", "best_params"], data=risk_table_rows)})
    run.finish()
