# src/regression_analyzer/stats/feature_importance.py

from typing import List
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .models import FeatureImportanceResult, FeatureImportance


def calculate_feature_importance(
    df: pl.DataFrame,
    target: str,
    features: List[str],
    n_estimators: int = 100,
    n_repeats: int = 10,
    random_state: int = 42
) -> FeatureImportanceResult:
    """Calculate feature importance using Random Forest + permutation importance.

    Uses permutation_importance instead of tree's feature_importances_
    because it's model-agnostic and unbiased toward high-cardinality features.

    Args:
        df: Input DataFrame
        target: Target column name
        features: Feature column names
        n_estimators: Number of trees in forest
        n_repeats: Permutation importance repeats
        random_state: Random seed

    Returns:
        FeatureImportanceResult with ranked features
    """
    # Prepare data
    df_clean = df.select([target] + features).drop_nulls()

    if df_clean.height < 20:
        raise ValueError(f"Insufficient data: {df_clean.height} rows after dropping nulls")

    X = df_clean.select(features).to_numpy()
    y = df_clean.select(target).to_numpy().flatten()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Fit Random Forest
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Model score
    r2 = model.score(X_test, y_test)

    # Permutation importance (on test set)
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Build results
    importance_scores = []
    for i, feature in enumerate(features):
        importance_scores.append({
            "feature": feature,
            "importance": float(perm_importance.importances_mean[i]),
            "importance_std": float(perm_importance.importances_std[i]),
        })

    # Sort by importance and add rank
    importance_scores.sort(key=lambda x: x["importance"], reverse=True)
    for rank, item in enumerate(importance_scores, 1):
        item["rank"] = rank

    feature_importances = [
        FeatureImportance(**item) for item in importance_scores
    ]

    # Generate interpretation
    interpretation = _interpret_importance(target, feature_importances, r2)

    return FeatureImportanceResult(
        target=target,
        features=feature_importances,
        model_r_squared=float(r2),
        method="permutation_importance",
        interpretation=interpretation
    )


def _interpret_importance(
    target: str,
    features: List[FeatureImportance],
    r2: float
) -> str:
    """Generate interpretation of feature importance."""
    lines = []

    lines.append(f"Random Forest model R²={r2:.2f}.")

    # Top features
    top_features = [f for f in features if f.importance > 0.01][:5]
    if top_features:
        lines.append(f"Top predictors of {target}:")
        for f in top_features:
            lines.append(f"  {f.rank}. {f.feature} (importance: {f.importance:.3f})")

    # Negligible features
    negligible = [f for f in features if f.importance < 0.001]
    if negligible:
        lines.append(f"{len(negligible)} features have negligible importance.")

    lines.append("Note: Importance indicates predictive power, not causation.")

    return " ".join(lines)
