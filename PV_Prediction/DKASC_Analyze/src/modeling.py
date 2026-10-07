from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def regression_metrics(y_true, y_pred, mape_threshold: float = 0.05) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    mape_mask = np.abs(y_true) > mape_threshold
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE": float(np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100)
        if mape_mask.any()
        else np.nan,
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }


def make_ridge_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_random_forest_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=18,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def deterministic_downsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, max_rows).astype(int)
    return df.iloc[idx]


def train_and_evaluate_models(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    random_state: int,
    max_rf_train_rows: int,
    mape_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    model_df = df[features + [target_col]].dropna(subset=[target_col]).copy()
    model_df = model_df.sort_index()
    train_df, val_df, test_df = chronological_split(model_df)
    models = {
        "Ridge": make_ridge_model(),
        "RandomForest": make_random_forest_model(random_state),
    }
    metrics_rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    predictions = pd.DataFrame(index=test_df.index)
    predictions["actual"] = test_df[target_col]

    for model_name, model in models.items():
        fit_df = train_df
        if model_name == "RandomForest":
            fit_df = deterministic_downsample(train_df, max_rf_train_rows)
        model.fit(fit_df[features], fit_df[target_col])
        fitted[model_name] = model
        for split_name, split_df in [("validation", val_df), ("test", test_df)]:
            y_pred = model.predict(split_df[features])
            metrics = regression_metrics(split_df[target_col], y_pred, mape_threshold=mape_threshold)
            metrics_rows.append({"model": model_name, "split": split_name, **metrics, "n": len(split_df)})
            if split_name == "test":
                predictions[f"predicted_{model_name}"] = y_pred

    rf = fitted["RandomForest"]
    rf_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": rf.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    perm_sample = deterministic_downsample(test_df, 30_000)
    perm = permutation_importance(
        rf,
        perm_sample[features],
        perm_sample[target_col],
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    perm_importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return pd.DataFrame(metrics_rows), fitted, rf_importance, perm_importance, predictions
