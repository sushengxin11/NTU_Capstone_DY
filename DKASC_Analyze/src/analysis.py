from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer


def correlation_table(df: pd.DataFrame, features: list[str], target_col: str) -> pd.DataFrame:
    available = [col for col in features if col in df.columns and col != target_col]
    rows: list[dict[str, object]] = []
    for col in available:
        pair = df[[col, target_col]].dropna()
        if len(pair) < 3:
            continue
        rows.append(
            {
                "feature": col,
                "pearson": pair[col].corr(pair[target_col], method="pearson"),
                "spearman": pair[col].corr(pair[target_col], method="spearman"),
                "n": len(pair),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_pearson"] = out["pearson"].abs()
        out["abs_spearman"] = out["spearman"].abs()
        out = out.sort_values("abs_pearson", ascending=False)
    return out


def mutual_information_table(df: pd.DataFrame, features: list[str], target_col: str, random_state: int) -> pd.DataFrame:
    available = [col for col in features if col in df.columns and col != target_col]
    model_df = df[available + [target_col]].dropna(subset=[target_col])
    if model_df.empty or not available:
        return pd.DataFrame(columns=["feature", "mutual_information"])
    x = model_df[available]
    y = model_df[target_col]
    x_imputed = SimpleImputer(strategy="median").fit_transform(x)
    scores = mutual_info_regression(x_imputed, y, random_state=random_state)
    return pd.DataFrame({"feature": available, "mutual_information": scores}).sort_values(
        "mutual_information", ascending=False
    )
