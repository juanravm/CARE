import argparse
import os
import joblib
import numpy as np
import pandas as pd

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


def load_inference_data(path: str) -> pd.DataFrame:
    """
    Load new data for inference.
    Assumes same columns as training; drops target columns if present.
    """
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    drop_cols = [c for c in ["risk_status", "dfs_status", "dfs_time", "os_status"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # Standard Scale numerical variables
    scaler = StandardScaler()
    cols = ["edad", "imc", "tamano_tumoral"]
    df[cols] = scaler.fit_transform(df[cols])
    return df


def predict_class_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run all classification models and return probabilities.
    """
    outputs = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            outputs[name] = prob
    return pd.DataFrame(outputs)


def predict_risk_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run survival/risk models.
    Output:
      - cox/xgb_cox: partial hazard (higher -> greater risk)
      - rsf: cumulative hazard at last time point (approx risk)
    """
    outputs = {}
    for name, model in models.items():
        if hasattr(model, "predict_cumulative_hazard_function"):
            # RandomSurvivalForest: take cumulative hazard at last time
            chfs = model.predict_cumulative_hazard_function(X)
            outputs[name] = np.array([fn.y[-1] for fn in chfs])
        elif hasattr(model, "predict"):
            outputs[name] = model.predict(X)
    return pd.DataFrame(outputs)


def main(args: argparse.Namespace):
    class_models, risk_models = load_models(args.models_dir)
    df = load_inference_data(args.data)

    class_preds = predict_class_models(class_models, df)
    risk_preds = predict_risk_models(risk_models, df)

    os.makedirs(args.output_dir, exist_ok=True)
    class_path = os.path.join(args.output_dir, "class_predictions.csv")
    risk_path = os.path.join(args.output_dir, "risk_scores.csv")

    if not class_preds.empty:
        class_preds.to_csv(class_path, index=False)
        print(f"Saved classification probabilities to {class_path}")
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
