import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


MODEL_FILE = "models/xgboost_neighbourhood_fixed_ura_best.pkl"
LABELS_CSV = "labels_real_zoning_10378.csv"
FEATURES_CSV = "features_with_neighbourhood.csv"
URA_ZONES_CSV = "building_zones_ura_FIXED.csv"
IMPORTANCE_CSV = "xgboost_feature_importance_fixed_ura_best.csv"

OUTPUT_DIR = "final_model_graphs"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_target_column(df):
    print("\nAvailable columns containing 'height':")
    height_cols = [c for c in df.columns if "height" in c.lower()]
    print(height_cols)

    preferred = [
        "height_labels",
        "height_label",
        "height",
        "height_x",
        "height_y",
    ]

    for col in preferred:
        if col in df.columns:
            print(f"Using target column: {col}")
            return col

    if height_cols:
        print(f"Using target column: {height_cols[0]}")
        return height_cols[0]

    raise ValueError("No height target column found.")


def load_data():
    print("Loading model...")
    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    print("Loading CSV files...")
    labels = pd.read_csv(LABELS_CSV)
    features = pd.read_csv(FEATURES_CSV)
    zones = pd.read_csv(URA_ZONES_CSV)

    df = labels.merge(
        features,
        on="building_id",
        how="inner",
        suffixes=("_labels", "_features")
    )

    df = df.merge(
        zones[["building_id", "ura_zone"]],
        on="building_id",
        how="left"
    )

    df["ura_zone"] = df["ura_zone"].fillna("Unmatched")

    target_col = find_target_column(df)

    ura_dummies = pd.get_dummies(
        df["ura_zone"].astype(str),
        prefix="ura"
    )

    df = pd.concat([df, ura_dummies], axis=1)

    missing_features = [c for c in feature_cols if c not in df.columns]

    if missing_features:
        print("\nMissing features from dataframe:")
        for c in missing_features:
            print(f"  - {c}")

        print("\nCreating missing features as 0 so graph generation can continue.")
        for c in missing_features:
            df[c] = 0

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(float)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_SEED
    )

    y_pred = model.predict(X_test)

    return y_test, y_pred


def plot_predicted_vs_actual(y_test, y_pred):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.4, s=18)

    min_v = min(float(y_test.min()), float(np.min(y_pred)))
    max_v = max(float(y_test.max()), float(np.max(y_pred)))

    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("Actual Height (m)")
    plt.ylabel("Predicted Height (m)")
    plt.title("Predicted vs Actual Building Heights")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "01_predicted_vs_actual.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_error_distribution(y_test, y_pred):
    errors = y_test.values - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.75)
    plt.axvline(0, linestyle="--")

    plt.xlabel("Prediction Error (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "02_error_distribution.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_absolute_error_distribution(y_test, y_pred):
    abs_errors = np.abs(y_test.values - y_pred)

    plt.figure(figsize=(8, 5))
    plt.hist(abs_errors, bins=50, edgecolor="black", alpha=0.75)

    plt.xlabel("Absolute Error (m)")
    plt.ylabel("Frequency")
    plt.title("Absolute Prediction Error Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "03_absolute_error_distribution.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_feature_importance():
    if not os.path.exists(IMPORTANCE_CSV):
        print(f"{IMPORTANCE_CSV} not found. Skipping feature importance plot.")
        return

    importance = pd.read_csv(IMPORTANCE_CSV)

    if "feature" not in importance.columns or "importance" not in importance.columns:
        print("Feature importance CSV does not contain feature/importance columns.")
        return

    top = importance.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 XGBoost Feature Importances")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "04_feature_importance.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_metrics_summary(y_test, y_pred):
    errors = np.abs(y_test.values - y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    max_error = errors.max()
    p90 = np.percentile(errors, 90)
    p50 = np.percentile(errors, 50)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "P90 Error": p90,
        "P50 Error": p50
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())

    plt.ylabel("Value")
    plt.title("Final Model Performance Summary")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "05_metrics_summary.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

    metrics_out = os.path.join(OUTPUT_DIR, "final_metrics_summary.json")

    full_metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "max_error": float(max_error),
        "p90_error": float(p90),
        "p50_error": float(p50),
        "num_test_samples": int(len(y_test))
    }

    with open(metrics_out, "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"Saved {metrics_out}")

    print("\nFinal Metrics")
    print("=" * 40)
    print(f"RMSE:      {rmse:.4f} m")
    print(f"MAE:       {mae:.4f} m")
    print(f"R²:        {r2:.4f}")
    print(f"Max Error: {max_error:.4f} m")
    print(f"P90 Error: {p90:.4f} m")
    print(f"P50 Error: {p50:.4f} m")


def main():
    y_test, y_pred = load_data()

    plot_predicted_vs_actual(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)
    plot_absolute_error_distribution(y_test, y_pred)
    plot_feature_importance()
    plot_metrics_summary(y_test, y_pred)

    print("\nDone. Check the final_model_graphs folder.")


if __name__ == "__main__":
    main()