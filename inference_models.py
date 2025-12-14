import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Import custom classes so that unpickling works
from src.architecture.models import StackingModel  # noqa: F401
from src.preproc.preproc_utils import FCBFSelector, ToCuPy  # noqa: F401


def load_models(models_dir: str = "artifacts/models"):
    """
    Load all persisted models from disk.
    Returns two dicts: class_models, risk_models.
    """
    class_models = {}
    risk_models = {}
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    for fname in os.listdir(models_dir):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(models_dir, fname)
        if fname.startswith("class_"):
            name = fname.removeprefix("class_").removesuffix(".pkl")
            class_models[name] = joblib.load(path)
        elif fname.startswith("risk_"):
            name = fname.removeprefix("risk_").removesuffix(".pkl")
            risk_models[name] = joblib.load(path)
    return class_models, risk_models


def load_inference_data(path: str):
    """
    Load new data for inference and, if available, ground-truth labels.
    Returns (X, y_class) where y_class is None when risk_status is absent.
    """
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    y_class = None
    if "risk_status" in df.columns:
        y_class = pd.to_numeric(df["risk_status"], errors="coerce")

    drop_cols = [c for c in ["risk_status", "dfs_status", "dfs_time", "os_status"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, y_class


def predict_class_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run all classification models and return probabilities.
    """
    outputs = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            outputs[name] = prob
    return pd.DataFrame(outputs, index=X.index)


def predict_risk_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run survival/risk models.
    Output:
      - cox/xgb_cox: partial hazard (higher -> greater risk)
      - rsf: cumulative hazard at last time point (approx risk)
    """
    outputs = {}
    for name, model in models.items():
        S_t0 = model.predict_survival_function(X)
        # Calculating the survival probability at 3 years time
        S_t0 = 1 - np.array([f(3 * 365) for f in S_t0])
        outputs[name] = S_t0
    return pd.DataFrame(outputs, index=X.index)


def evaluate_class_models(class_preds: pd.DataFrame, y_true: pd.Series, output_dir: str):
    """
    Compute AUC + best-sensitivity/specificity (Youden) and plot ROC curves.
    Saves a CSV with metrics and a PNG plot inside output_dir.
    """

    mask = y_true.notna()
    y_clean = y_true.loc[mask]
    preds_clean = class_preds.loc[mask]

    metrics = []
    plt.figure(figsize=(8, 6))
    for name, probs in preds_clean.items():
        auc = roc_auc_score(y_clean, probs)
        fpr, tpr, thresholds = roc_curve(y_clean, probs)

        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_threshold = thresholds[best_idx]
        sensitivity = tpr[best_idx]
        specificity = 1 - fpr[best_idx]

        metrics.append(
            {
                "model": name,
                "auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "threshold": best_threshold,
            }
        )

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.scatter(fpr[best_idx], tpr[best_idx], s=25, alpha=0.8)

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Azar")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Roc Curves - Classification models")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "class_roc_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    metrics_df = pd.DataFrame(metrics).sort_values(by="auc", ascending=False)
    metrics_path = os.path.join(output_dir, "class_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("Resumen AUC/sensibilidad/especificidad:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Métricas de clasificación guardadas en {metrics_path}")
    print(f"Curvas ROC guardadas en {plot_path}")
    return metrics_df


def main(args: argparse.Namespace):
    class_models, risk_models = load_models(args.models_dir)
    df, y_true = load_inference_data(args.data)

    class_preds = predict_class_models(class_models, df)
    risk_preds = predict_risk_models(risk_models, df)

    os.makedirs(args.output_dir, exist_ok=True)
    class_path = os.path.join(args.output_dir, "class_predictions.csv")
    risk_path = os.path.join(args.output_dir, "risk_scores.csv")

    if not class_preds.empty:
        class_preds.to_csv(class_path, index=False)
        print(f"Saved classification probabilities to {class_path}")
        evaluate_class_models(class_preds, y_true, args.output_dir)
    if not risk_preds.empty:
        risk_preds.to_csv(risk_path, index=False)
        print(f"Saved survival risk scores to {risk_path}")

    print("Risk model outputs:")
    print("- Cox / XGB Cox: higher partial hazard => mayor riesgo/peor pronóstico.")
    print("- RSF: valor de hazard acumulado al último tiempo (mayor => mayor riesgo).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with saved models.")
    parser.add_argument("--data", required=True, help="Path to inference data (CSV/TSV) with same features as training.")
    parser.add_argument("--models_dir", default="artifacts/models", help="Directory with saved .pkl models.")
    parser.add_argument("--output_dir", default="artifacts/preds", help="Where to save predictions.")
    main(parser.parse_args())
