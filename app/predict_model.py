#!/usr/bin/env python3

import sys
import random
import pandas as pd


def compute_score(row):
    """
    Compute fake risk score for a single patient row
    """
    score = (
        0.01 * row["edad"] +
        0.02 * row["tamano_tumoral"] +
        0.015 * row["imc"] +
        random.uniform(-0.05, 0.05)
    )

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    group = "High" if row["tamano_tumoral"] > 5 else "Low"
    return score, group


def main():
    if len(sys.argv) != 2:
        print("Usage: predict_model.py <input.tsv> ", file=sys.stderr)
        sys.exit(1)

    input_tsv = sys.argv[1]
    output_tsv = '/home/natasamortvanski@vhio.org/CARE/app/tmp_files/output_predictions.tsv'

    # --------------------------------------------------
    # Read input TSV
    # --------------------------------------------------
    try:
        df = pd.read_csv(input_tsv, sep="\t")
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"edad", "tamano_tumoral", "imc"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Ensure Patient_ID exists
    if "Patient_ID" not in df.columns:
        df["Patient_ID"] = [f"Patient_{i+1}" for i in range(len(df))]

    # --------------------------------------------------
    # Compute predictions per patient
    # --------------------------------------------------
    results = []

    for _, row in df.iterrows():
        score, group = compute_score(row)
        results.append({
            "Patient_ID": row["Patient_ID"],
            "Score": round(score, 4),
            "Group": group
        })

    result_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Write output TSV
    # --------------------------------------------------
    result_df.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    main()
