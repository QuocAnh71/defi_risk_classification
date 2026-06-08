"""
Feature engineering module for the risk assessment pipeline.

This module provides purely functional, stateless transformations for feature
selection, outlier clamping, log1p transformations, and integrity checks using
native Polars expressions.
"""

import polars as pl
from pathlib import Path
from typing import Dict, List

HEAVY_TAIL_KEYWORDS: list[str] = ["eth", "gas", "amount", "count", "balance"]


def load_feature_contract(shortlist_path: str) -> list[str]:
    """
    Load the list of feature names to keep from a CSV file.

    Args:
        shortlist_path (str): The path to the CSV file containing the shortlist.

    Returns:
        list[str]: A list of feature names where the status is "KEEP".

    Raises:
        FileNotFoundError: If the specified shortlist path does not exist.

    Example:
        keep_features = load_feature_contract("reports/model_feature_shortlist.csv")
    """
    path = Path(shortlist_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature shortlist file not found at: {shortlist_path}")

    df = pl.read_csv(shortlist_path)
    keep_features = df.filter(pl.col("status") == "KEEP").select("Feature_Name").to_series().to_list()
    
    print(f"[CONTRACT] Loaded {len(keep_features)} KEEP features from {shortlist_path}")
    return keep_features


def apply_filter_contract(
    df: pl.DataFrame,
    keep_features: list[str],
    label_col: str = "target",
    split_tag: str = "data"
) -> pl.DataFrame:
    """
    Filter the DataFrame to only include the kept features and the label column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        keep_features (list[str]): The list of feature names to keep.
        label_col (str, optional): The name of the label/target column. Defaults to "target".
        split_tag (str, optional): A tag identifying the split for logging. Defaults to "data".

    Returns:
        pl.DataFrame: The filtered DataFrame containing only keep_features and label_col.

    Raises:
        ValueError: If any of the features in keep_features are missing from the DataFrame.
        AssertionError: If the resulting DataFrame does not have the expected number of columns.

    Example:
        filtered_df = apply_filter_contract(df, keep_features)
    """
    missing_cols = [f for f in keep_features if f not in df.columns]
    if missing_cols:
        raise ValueError(f"The following keep_features are missing from the DataFrame: {missing_cols}")
    
    final_cols = keep_features + [label_col]
    original_cols_count = df.shape[1]
    
    filtered_df = df.select(final_cols)
    
    expected_cols = len(keep_features) + 1
    if filtered_df.shape[1] != expected_cols:
        raise AssertionError(f"Expected {expected_cols} columns, got {filtered_df.shape[1]}")
    
    print(f"[FILTER] {split_tag} : {original_cols_count} cols -> {filtered_df.shape[1]} cols")
    return filtered_df


def fit_outlier_bounds(
    train_df: pl.DataFrame,
    numeric_cols: list[str],
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> dict[str, dict[str, float]]:
    """
    Compute lower and upper quantile bounds for numeric columns based on training data.

    Args:
        train_df (pl.DataFrame): The training DataFrame.
        numeric_cols (list[str]): List of numeric column names to compute bounds for.
        lower_pct (float, optional): The lower quantile percentage. Defaults to 0.01.
        upper_pct (float, optional): The upper quantile percentage. Defaults to 0.99.

    Returns:
        dict[str, dict[str, float]]: A dictionary mapping column names to their computed lower and upper bounds.

    Example:
        bounds = fit_outlier_bounds(train_df, ["amount", "balance"])
    """
    bounds: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        lower_bound = train_df.select(pl.col(col).quantile(lower_pct, interpolation="linear")).item()
        upper_bound = train_df.select(pl.col(col).quantile(upper_pct, interpolation="linear")).item()
        bounds[col] = {"lower": float(lower_bound), "upper": float(upper_bound)}
        
    print(f"[OUTLIER FIT] Computed P1/P99 bounds for {len(numeric_cols)} numeric columns (train only)")
    return bounds


def apply_clamping(
    df: pl.DataFrame,
    bounds: dict[str, dict[str, float]],
    split_tag: str = "data",
) -> pl.DataFrame:
    """
    Apply clipping to columns based on pre-computed lower and upper bounds.

    Args:
        df (pl.DataFrame): The input DataFrame.
        bounds (dict[str, dict[str, float]]): The dictionary of column bounds.
        split_tag (str, optional): A tag identifying the split for logging. Defaults to "data".

    Returns:
        pl.DataFrame: The clamped DataFrame.

    Example:
        clamped_df = apply_clamping(df, bounds)
    """
    clamping_exprs = []
    applied_cols = 0
    
    for col_name in df.columns:
        if col_name in bounds:
            clamping_exprs.append(
                pl.col(col_name).clip(
                    lower_bound=bounds[col_name]["lower"],
                    upper_bound=bounds[col_name]["upper"]
                )
            )
            applied_cols += 1
            
    if clamping_exprs:
        df = df.with_columns(clamping_exprs)
        
    print(f"[CLAMP] {split_tag} : applied clamping to {applied_cols} columns")
    return df


def apply_log1p_transform(
    df: pl.DataFrame,
    split_tag: str = "data",
) -> pl.DataFrame:
    """
    Apply log1p transformation to numeric columns matching specific keywords.

    Args:
        df (pl.DataFrame): The input DataFrame.
        split_tag (str, optional): A tag identifying the split for logging. Defaults to "data".

    Returns:
        pl.DataFrame: The transformed DataFrame.

    Example:
        log1p_df = apply_log1p_transform(df)
    """
    valid_dtypes = {pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64}
    transform_exprs = []
    transformed_cols = []
    
    for col_name in df.columns:
        if any(keyword in col_name.lower() for keyword in HEAVY_TAIL_KEYWORDS):
            if df.schema[col_name] in valid_dtypes:
                transform_exprs.append(pl.col(col_name).log1p())
                transformed_cols.append(col_name)
                
    if transform_exprs:
        df = df.with_columns(transform_exprs)
        
    print(f"[LOG1P] {split_tag} : applied log1p to {len(transformed_cols)} columns -> {transformed_cols}")
    return df


def assert_feature_integrity(
    df: pl.DataFrame,
    expected_rows: int,
    expected_cols: int = 53,
    split_tag: str = "data",
) -> None:
    """
    Perform structural and integrity assertions on the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        expected_rows (int): The expected number of rows.
        expected_cols (int, optional): The expected number of columns. Defaults to 53.
        split_tag (str, optional): A tag identifying the split for logging. Defaults to "data".

    Raises:
        AssertionError: If any of the integrity checks fail.

    Example:
        assert_feature_integrity(df, 1000)
    """
    # CHECK 1: Row count
    rows = df.shape[0]
    if rows == expected_rows:
        print(f"[ASSERT] {split_tag} rows   : PASS ({rows} rows)")
    else:
        print(f"[ASSERT] {split_tag} rows   : FAIL (expected {expected_rows}, got {rows})")
        raise AssertionError(f"Expected {expected_rows} rows, got {rows}")

    # CHECK 2: Column count
    cols = df.shape[1]
    if cols == expected_cols:
        print(f"[ASSERT] {split_tag} cols   : PASS ({cols} cols)")
    else:
        print(f"[ASSERT] {split_tag} cols   : FAIL (expected {expected_cols}, got {cols})")
        raise AssertionError(f"Expected {expected_cols} cols, got {cols}")

    # CHECK 3: Zero nulls
    total_null = df.null_count().sum_horizontal().item()
        
    if total_null == 0:
        print(f"[ASSERT] {split_tag} nulls  : PASS (0 nulls)")
    else:
        print(f"[ASSERT] {split_tag} nulls  : FAIL (found {total_null} nulls)")
        raise AssertionError(f"Expected 0 nulls, found {total_null}")

    # CHECK 4: Target column present
    if "target" in df.columns:
        print(f"[ASSERT] {split_tag} target : PASS (column exists)")
    else:
        print(f"[ASSERT] {split_tag} target : FAIL (column missing)")
        raise AssertionError("Target column is missing")

    print(f"[INTEGRITY] {split_tag} : ALL CHECKS PASSED")
    print("================================================")
