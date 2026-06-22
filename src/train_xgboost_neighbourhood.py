import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


LABELS_CSV = "labels_real_zoning_10378.csv"
FEATURES_CSV = "features_with_neighbourhood.csv"
URA_ZONES_CSV = "building_zones_ura_FIXED.csv"

MODEL_OUT = "models/xgboost_neighbourhood_fixed_ura_best.pkl"
METRICS_OUT = "eval_metrics_xgboost_neighbourhood_fixed_ura_best.json"
IMPORTANCE_OUT = "xgboost_feature_importance_fixed_ura_best.csv"

RANDOM_SEED = 42


def evaluate(y_true, y_pred):
    errors = np.abs(y_true.values - y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "max_error": float(errors.max()),
        "p90_error": float(np.percentile(errors, 90)),
        "p50_error": float(np.percentile(errors, 50)),
    }


def main():
    os.makedirs("models", exist_ok=True)

    df_labels = pd.read_csv(LABELS_CSV)
    df_features = pd.read_csv(FEATURES_CSV)
    df_zones = pd.read_csv(URA_ZONES_CSV)

    df = df_labels.merge(
        df_features,
        on="building_id",
        how="inner",
        suffixes=("_labels", "_features")
    )

    df = df.merge(
        df_zones[["building_id", "ura_zone"]],
        on="building_id",
        how="left"
    )

    df["ura_zone"] = df["ura_zone"].fillna("Unmatched")

    print(f"Dataset size: {len(df)}")

    target_col = None
    for col in df.columns:
        if "height_labels" in col:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Could not find target height column.")

    print(f"Using target column: {target_col}")

    print("\nCorrected URA Zone Distribution")
    print("=" * 50)
    print(df["ura_zone"].value_counts().head(20))

    exclude_cols = [
        "building_id",
        "mask_path",
        "mask_path_labels",
        "zone_class",
        "zoning",
        "ura_zone",
        "latitude",
        "longitude",
        "x",
        "y",
        "z",
    ]

    exclude_cols += [
        col for col in df.columns
        if "height" in col.lower()
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    ura_dummies = pd.get_dummies(
        df["ura_zone"].astype(str),
        prefix="ura"
    )

    df = pd.concat([df, ura_dummies], axis=1)
    feature_cols += list(ura_dummies.columns)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_SEED
    )

    configs = [
        {
            "name": "baseline_best_previous",
            "n_estimators": 1000,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "min_child_weight": 1,
        },
        {
            "name": "deeper_more_trees",
            "n_estimators": 1600,
            "max_depth": 6,
            "learning_rate": 0.02,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.05,
            "min_child_weight": 1,
        },
        {
            "name": "medium_regularized",
            "n_estimators": 1800,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "reg_alpha": 0.1,
            "min_child_weight": 2,
        },
        {
            "name": "shallow_stable",
            "n_estimators": 2200,
            "max_depth": 4,
            "learning_rate": 0.015,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.5,
            "reg_alpha": 0.05,
            "min_child_weight": 1,
        },
        {
            "name": "deeper_low_lr",
            "n_estimators": 2500,
            "max_depth": 7,
            "learning_rate": 0.01,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "reg_alpha": 0.05,
            "min_child_weight": 1,
        },
    ]

    best_model = None
    best_metrics = None
    best_config = None
    all_results = []

    for cfg in configs:
        print("\n" + "=" * 70)
        print(f"Training config: {cfg['name']}")
        print("=" * 70)

        model = XGBRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg["reg_lambda"],
            reg_alpha=cfg["reg_alpha"],
            min_child_weight=cfg["min_child_weight"],
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="rmse",
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["config"] = cfg["name"]
        all_results.append(metrics)

        print(f"\n{cfg['name']} results")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_model = model
            best_metrics = metrics
            best_config = cfg

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(best_config)
    print(f"RMSE:      {best_metrics['rmse']:.4f} m")
    print(f"MAE:       {best_metrics['mae']:.4f} m")
    print(f"R²:        {best_metrics['r2']:.4f}")
    print(f"Max Error: {best_metrics['max_error']:.4f} m")
    print(f"P90 Error: {best_metrics['p90_error']:.4f} m")
    print(f"P50 Error: {best_metrics['p50_error']:.4f} m")

    final_metrics = {
        "model": "Best XGBoost + corrected URA zoning + neighbourhood features",
        "best_config": best_config,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "num_test_samples": int(len(y_test)),
        "num_features": int(len(feature_cols)),
        "features": feature_cols,
    }

    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "best_config": best_config,
        },
        MODEL_OUT
    )

    with open(METRICS_OUT, "w") as f:
        json.dump(final_metrics, f, indent=2)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(IMPORTANCE_OUT, index=False)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    print(f"Saved feature importance to {IMPORTANCE_OUT}")

    print("\nTop 20 Feature Importances")
    print(importance.head(20))


if __name__ == "__main__":
    main()