"""
Preprocessing module for DeFi credit risk assessment.
Serves as the single source of truth for all data-cleaning logic.
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple

def load_raw_data(path: str) -> pl.DataFrame:
    """
    Load the raw parquet dataset using Polars.

    Args:
        path: Relative or absolute path to the .parquet file.

    Returns:
        pl.DataFrame of the raw dataset.
    """
    df = pl.read_parquet(path)
    
    print(f"File path loaded: {path}")
    print(f"Shape: {df.height} x {df.width}")
    print(f"Memory usage: {df.estimated_size('mb'):.2f} MB")
    
    return df

def audit_nulls(df: pl.DataFrame) -> Dict[str, int]:
    """
    Scan every column for null/NaN values.

    Returns:
        A dict mapping column_name -> null_count, FILTERED to only include
        columns where null_count > 0. Returns an empty dict {} if the
        dataset is fully clean.
    """
    print(f"Total columns scanned: {df.width}")
    
    null_counts = df.null_count()
    result = {}
    for col in null_counts.columns:
        val = null_counts[col][0]
        if val > 0:
            result[col] = val
            
    if not result:
        print("NULL AUDIT PASSED: 0 null values found across all columns.")
    else:
        for col, count in result.items():
            print(f"  {col}: {count} nulls")
            
    return result

def apply_median_imputation(df: pl.DataFrame, null_cols: List[str]) -> pl.DataFrame:
    """
    Fill null values in the given columns with per-column median.
    Only call this function when audit_nulls returns a non-empty dict.

    Args:
        df:        The input DataFrame (may contain nulls).
        null_cols: List of column names that have nulls (from audit_nulls keys).

    Returns:
        New pl.DataFrame with nulls replaced. Original df is NOT mutated.
    """
    exprs = [pl.col(c).fill_null(pl.col(c).median()) for c in null_cols]
    
    print(f"IMPUTATION APPLIED: median fill on {len(null_cols)} column(s): {null_cols}")
    
    return df.with_columns(exprs)

def prune_flash_wallets(df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
    """
    Remove behavioral noise: wallets that are ephemeral/flash accounts.

    A row is classified as a flash wallet and REMOVED if ALL four conditions
    hold simultaneously:
        - wallet_age == 0
        - incoming_tx_count <= 1
        - total_collateral_eth == 0
        - borrow_amount_sum_eth == 0

    Returns:
        Tuple of (cleaned_df, n_removed) where n_removed is the count of
        rows pruned.
    """
    mask = (
        (pl.col("wallet_age") == 0) &
        (pl.col("incoming_tx_count") <= 1) &
        (pl.col("total_collateral_eth") == 0) &
        (pl.col("borrow_amount_sum_eth") == 0)
    )
    
    cleaned_df = df.filter(~mask)
    
    n_before = df.height
    n_after = cleaned_df.height
    n_removed = n_before - n_after
    pct_removed = (n_removed / n_before * 100) if n_before > 0 else 0.0
    
    print("FLASH WALLET PRUNING:")
    print(f"  Rows before : {n_before}")
    print(f"  Rows removed: {n_removed} ({pct_removed:.2f}%)")
    print(f"  Rows after  : {n_after}")
    
    return cleaned_df, n_removed

def temporal_split(df: pl.DataFrame, timestamp_col: str, cutoff_ts: int = 1672531200) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split the dataset into train and test sets using a strict temporal
    boundary. No row may appear in both sets (wall-clock split, not random).

    Args:
        df:            Input DataFrame (after flash wallet pruning).
        timestamp_col: Name of the Unix timestamp column to split on.
        cutoff_ts:     Unix timestamp of the boundary (default = 1672531200,
                       which is 2023-01-01 00:00:00 UTC).

    Returns:
        Tuple (train_df, test_df) where:
            train_df: rows where timestamp_col  < cutoff_ts  (pre-2023)
            test_df:  rows where timestamp_col >= cutoff_ts  (post-2023)
    """
    train_df = df.filter(pl.col(timestamp_col) < cutoff_ts)
    test_df = df.filter(pl.col(timestamp_col) >= cutoff_ts)
    
    total_rows = df.height
    train_count = train_df.height
    test_count = test_df.height
    
    train_pct = (train_count / total_rows * 100) if total_rows > 0 else 0.0
    test_pct = (test_count / total_rows * 100) if total_rows > 0 else 0.0
    
    print(f"TEMPORAL SPLIT at cutoff = {cutoff_ts} (2023-01-01 UTC):")
    print(f"  Train (pre-2023) : {train_count} rows  ({train_pct:.1f}%)")
    print(f"  Test  (post-2023): {test_count} rows  ({test_pct:.1f}%)")
    
    if "target" in train_df.columns:
        vc = train_df.get_column("target").value_counts()
        train_targets = dict(zip(vc[vc.columns[0]], vc[vc.columns[1]]))
        train_0 = train_targets.get(0, 0)
        train_1 = train_targets.get(1, 0)
        train_ratio = (train_1 / train_0) if train_0 > 0 else 0.0
        print(f"  Target distribution in train -> 0: {train_0}  1: {train_1}  ratio: {train_ratio:.2f}")
    
    if "target" in test_df.columns:
        vc = test_df.get_column("target").value_counts()
        test_targets = dict(zip(vc[vc.columns[0]], vc[vc.columns[1]]))
        test_0 = test_targets.get(0, 0)
        test_1 = test_targets.get(1, 0)
        test_ratio = (test_1 / test_0) if test_0 > 0 else 0.0
        print(f"  Target distribution in test  -> 0: {test_0}  1: {test_1}  ratio: {test_ratio:.2f}")
        
    return train_df, test_df

def save_parquet(df: pl.DataFrame, path: str) -> None:
    """
    Persist a DataFrame to a compressed Parquet file.

    Args:
        df:   DataFrame to save.
        path: Destination file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.write_parquet(path, compression="snappy")
    
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"SAVED: {path}  |  rows={df.height}  cols={df.width}  size={size_mb:.2f} MB")
