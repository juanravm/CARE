import json
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer

from src.architecture.models import make_class_models, make_risk_models
from src.preproc.preproc_utils import FCBFSelector
from src.utils.utils import setup_hyperparameters

# 0) Setup run hyperparameters and W&B run
config_path = "src/configs/config.yaml"
_, run = setup_hyperparameters(config_path)

# 1) Loading data
df = pd.read_csv("data/training_df.csv")
target = ["DFS"]  # pon aquí tu variable objetivo
y = df[target].copy()
X = df.drop(columns=target)

# 2) Define categorical and numerical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# 4) Selector for best features
# selector = SelectKBest(score_func=f_classif, k=50)  # ajusta k
selector = FCBFSelector(mode="rank", threshold=0.0, n_bins=2)

# 5) Models and hyperparameters search
class_models = make_class_models()
risk_models = make_risk_models()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")

# 6) Class models
class_results = {}
class_table_rows = []
for name, (estimator, search_space) in class_models.items():
    pipe = Pipeline(
        [
            ("selector", selector),
            ("model", estimator),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions=search_space,
        n_iter=50,
        cv=cv,
        scoring=scorer,
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
        class_table_rows.append(
            [name, search.best_score_, json.dumps(search.best_params_, default=str)]
        )

# 7) Risk models
risk_results = {}
risk_table_rows = []
for name, (estimator, search_space) in risk_models.items():
    pipe = Pipeline(
        [
            ("selector", selector),
            ("model", estimator),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions=search_space,
        n_iter=50,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        random_state=0,
        refit=True,
    )
    search.fit(X, y)
    risk_results[name] = {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }
    print(f"{name}: AUC={search.best_score_:.3f}")
    if run:
        run.log(
            {
                f"risk/{name}/best_auc": search.best_score_,
                f"risk/{name}/best_params": json.dumps(search.best_params_, default=str),
            }
        )
        risk_table_rows.append(
            [name, search.best_score_, json.dumps(search.best_params_, default=str)]
        )

if run:
    if class_table_rows:
        run.log(
            {"class_results": wandb.Table(columns=["model", "best_auc", "best_params"], data=class_table_rows)}
        )
    if risk_table_rows:
        run.log(
            {"risk_results": wandb.Table(columns=["model", "best_auc", "best_params"], data=risk_table_rows)}
        )
    run.finish()
