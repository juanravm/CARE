import json
import os
import pandas as pd
import wandb
import joblib
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

from src.architecture.models import make_class_models, make_risk_models
from src.preproc.preproc_utils import FCBFSelector, ToCuPy
from src.utils.utils import setup_hyperparameters
from src.architecture.models import StackingModel

# 0) Setup run hyperparameters and W&B run
config_path = "src/configs/config.yaml"
_, run = setup_hyperparameters(config_path)

# 1) Loading data
df = pd.read_csv("data/data_train.tsv", sep="\t")

# Filtering variables with too many missing values
na_frac = df.drop(columns=["risk_status", "dfs_status", "dfs_time"]).isna().mean()
keep = na_frac[na_frac <= 0.10].index.tolist()
df = df.loc[:, keep + ["risk_status", "dfs_status", "dfs_time"]]

# Removing samples with any missing value
keep = df.drop(columns=["risk_status", "dfs_status", "dfs_time"]).dropna().index.tolist()
df = df.loc[keep, :]

print(f"Training with {df.shape[0]} samples")


y_class = df["risk_status"].copy()
y_event = df["dfs_status"].copy()
y_time = df["dfs_time"].copy()
X_full = df.drop(columns=["risk_status", "dfs_status", "dfs_time", "os_status"])

scale_cols = [c for c in ["edad", "imc", "tamano_tumoral"] if c in df.columns]
if scale_cols:
    preprocessor = ColumnTransformer(
        [("num_scaler", StandardScaler(), scale_cols)],
        remainder="passthrough",
        sparse_threshold=0.0,
    )
else:
    preprocessor = "passthrough"


# Split para clasificación usando solo filas con y_class no nulo (se excluyen solo en el test/train de clasif)
mask_class = y_class.notna()
df_class = df.loc[mask_class].copy()
y_class_clean = y_class.loc[mask_class]

X_class_all = df_class.drop(columns=["risk_status"])
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_class_all,
    y_class_clean,
    test_size=0.2,
    stratify=y_class_clean,
    random_state=0,
)

# Guardar test set limpio (sin NAs) para clasificación/riesgo
test_mask = df.index.isin(X_test_cls.index)
X_test_cls["risk_status"] = y_test_cls
X_test_cls.to_csv("/home/juanrafaelvalera@vhio.org/ondemand/CARE/data/test_data.tsv", sep="\t")

# Datos de entrenamiento para clasificación
X_class = X_train_cls.drop(columns=["dfs_status", "dfs_time", "os_status"], errors="ignore")
y_class = y_train_cls.copy()

# Datos para riesgo: mantener pacientes con y_class NaN si tienen eventos/tiempos
mask_risk = y_event.notna() & y_time.notna()
risk_train_mask = mask_risk & ~test_mask

X_surv = X_full.loc[risk_train_mask]
y_time = y_time.loc[risk_train_mask]
y_event = y_event.loc[risk_train_mask]
y_surv = Surv.from_arrays(event=y_event.astype(bool), time=y_time)

# 4) Selector for best features
selector = FCBFSelector(mode="rank", threshold=0.0, n_bins=2)

# 5) Models and hyperparameters search
class_models = make_class_models()
risk_models = make_risk_models()

cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorer_class = make_scorer(roc_auc_score)
cv_risk = KFold(n_splits=5, shuffle=True, random_state=0)


def cindex_scorer(estimator, X, y_surv):
    pred = estimator.predict(X)
    return concordance_index_censored(y_surv["event"], y_surv["time"], pred)[0]


scorer_risk = cindex_scorer

# 6) Class models
class_results = {}
class_table_rows = []
# y_class = y_class.astype(int).to_numpy().ravel()

for name, (estimator, search_space) in class_models.items():
    if name == "xgb":
        pipe = Pipeline(
            [
                ("preprocess", preprocessor),
                ("selector", selector),
                ("tocupy", ToCuPy()),
                ("model", estimator),
            ]
        )
    else:
        pipe = Pipeline(
            [
                ("preprocess", preprocessor),
                ("selector", selector),
                ("model", estimator),
            ]
        )

    search = RandomizedSearchCV(
        pipe,
        param_distributions=search_space,
        n_iter=50,
        cv=cv_class,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=0,
        refit=True,
    )

    search.fit(X_class, y_class)
    class_results[name] = {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }
    print(f"{name}: AUC={search.best_score_:.3f}")
    class_table_rows.append([name, search.best_score_, json.dumps(search.best_params_, default=str)])
    if run:
        run.log(
            {
                f"class/{name}/best_auc": search.best_score_,
                f"class/{name}/best_params": json.dumps(search.best_params_, default=str),
            }
        )

# 6b) Stacking model con los mejores logreg, svc y rf
required_base = ["logreg", "svc", "rf"]
if all(k in class_results for k in required_base):
    stack_model = StackingModel(
        class_results["logreg"]["best_estimator"],
        class_results["svc"]["best_estimator"],
        class_results["rf"]["best_estimator"],
    )
    stack_model.fit(X_class, y_class)
    stack_auc = roc_auc_score(y_class, stack_model.predict_proba(X_class)[:, 1])
    class_results["stacking"] = {
        "best_score": stack_auc,
        "best_params": {"bases": required_base},
        "best_estimator": stack_model,
    }
    print(f"stacking(logreg,svc,rf): AUC={stack_auc:.3f}")
    class_table_rows.append(["stacking(logreg,svc,rf)", stack_auc, json.dumps({"bases": required_base})])
    if run:
        run.log({"class/stacking/best_auc": stack_auc})

# 7) Risk models
risk_results = {}
risk_table_rows = []

for name, (estimator, search_space) in risk_models.items():

    # Pipeline
    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("selector", selector),
            ("model", estimator),
        ]
    )

    # Caso 2: con hiperparámetros -> RandomizedSearchCV
    # Caso especial: XGB survival:cox no acepta y_surv (structured)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=search_space,
        n_iter=50,
        cv=cv_risk,
        scoring=scorer_risk,
        n_jobs=-1,
        random_state=0,
        refit=True,
        error_score="raise",
    )
    search.fit(X_surv, y_surv)

    best_score = float(search.best_score_)
    best_params = search.best_params_
    best_estimator = search.best_estimator_
    risk_table_rows.append([name, best_score, json.dumps(best_params, default=str)])

    # Guardar resultados
    risk_results[name] = {
        "best_score": best_score,
        "best_params": best_params,
        "best_estimator": best_estimator,
    }
    print(f"{name}: C-index={best_score:.3f}")

    if run:
        run.log(
            {
                f"risk/{name}/best_cindex": best_score,
                f"risk/{name}/best_params": json.dumps(best_params, default=str),
            }
        )

if run:
    if class_table_rows:
        run.log({"class_results": wandb.Table(columns=["model", "best_auc", "best_params"], data=class_table_rows)})
    if risk_table_rows:
        run.log({"risk_results": wandb.Table(columns=["model", "best_cindex", "best_params"], data=risk_table_rows)})
    run.finish()

# 7b) Persist cross-validation metrics locally
metrics_dir = "artifacts/metrics"
os.makedirs(metrics_dir, exist_ok=True)

if class_table_rows:
    class_df = pd.DataFrame(class_table_rows, columns=["model", "best_auc", "best_params"])
    class_metrics_path = os.path.join(metrics_dir, "class_cv_results.csv")
    class_df.to_csv(class_metrics_path, index=False)
    print(f"Resultados de AUC de clasificación guardados en {class_metrics_path}")

if risk_table_rows:
    risk_df = pd.DataFrame(risk_table_rows, columns=["model", "best_cindex", "best_params"])
    risk_metrics_path = os.path.join(metrics_dir, "risk_cv_results.csv")
    risk_df.to_csv(risk_metrics_path, index=False)
    print(f"Resultados de C-index de riesgo guardados en {risk_metrics_path}")

# 8) Persist best estimators for inference
models_dir = "artifacts/models"
os.makedirs(models_dir, exist_ok=True)

for name, result in class_results.items():
    joblib.dump(result["best_estimator"], os.path.join(models_dir, f"class_{name}.pkl"))

for name, result in risk_results.items():
    joblib.dump(result["best_estimator"], os.path.join(models_dir, f"risk_{name}.pkl"))
