"""
Data validation utilities for Economic Trend Forecasting System.
Add this as modules/data_validation.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data quality checks fail."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
    required_columns: List[str] = None,
    check_duplicates: bool = True,
    check_nulls: bool = True,
    check_numeric: bool = False,
) -> Tuple[bool, Dict]:
    """
    Validate dataframe quality.
    
    Args:
        df: Input dataframe
        name: Name for logging
        required_columns: List of required columns
        check_duplicates: Check for duplicate rows
        check_nulls: Check for null values
        check_numeric: Check if all values are numeric
    
    Returns:
        (is_valid, report_dict)
    """
    report = {
        "name": name,
        "shape": df.shape,
        "issues": [],
        "warnings": [],
    }
    
    # Check if empty
    if df.empty:
        report["issues"].append("DataFrame is empty")
        logger.error(f"{name} is empty")
        return False, report
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            report["issues"].append(f"Missing columns: {missing}")
            logger.error(f"{name} missing columns: {missing}")
            return False, report
    
    # Check for null values
    if check_nulls:
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_info = null_counts[null_counts > 0].to_dict()
            if null_counts.sum() / len(df) > 0.1:  # >10% nulls
                report["issues"].append(f"High null values: {null_info}")
                logger.error(f"{name} has high null values: {null_info}")
                return False, report
            else:
                report["warnings"].append(f"Some null values: {null_info}")
                logger.warning(f"{name} has some nulls: {null_info}")
    
    # Check for duplicates
    if check_duplicates:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = (dup_count / len(df)) * 100
            if pct > 5:  # >5% duplicates
                report["issues"].append(f"High duplicate rows: {dup_count} ({pct:.1f}%)")
                logger.error(f"{name} has {dup_count} duplicate rows")
                return False, report
            else:
                report["warnings"].append(f"Some duplicate rows: {dup_count} ({pct:.1f}%)")
                logger.warning(f"{name} has {dup_count} duplicates")
    
    # Check if numeric
    if check_numeric:
        for col in df.select_dtypes(exclude=['number']).columns:
            try:
                pd.to_numeric(df[col], errors='coerce')
                pct_converted = df[col].astype(str).str.isnumeric().sum() / len(df)
                if pct_converted < 0.8:  # <80% numeric
                    report["issues"].append(f"Column '{col}' is not sufficiently numeric ({pct_converted:.1%})")
                    logger.error(f"{name}: Column '{col}' is not numeric enough")
                    return False, report
            except Exception as e:
                report["warnings"].append(f"Could not validate numeric on '{col}': {e}")
    
    logger.info(f"{name} validation passed. Shape: {df.shape}")
    return True, report


def validate_price_series(
    prices: pd.Series,
    name: str = "Price Series",
    min_length: int = 30,
) -> Tuple[bool, Dict]:
    """
    Validate price time series.
    
    Args:
        prices: Price series
        name: Name for logging
        min_length: Minimum required length
    
    Returns:
        (is_valid, report_dict)
    """
    report = {
        "name": name,
        "length": len(prices),
        "issues": [],
        "warnings": [],
    }
    
    # Check length
    if len(prices) < min_length:
        report["issues"].append(f"Series too short: {len(prices)} < {min_length}")
        logger.error(f"{name} is too short: {len(prices)} values")
        return False, report
    
    # Check for nulls
    null_count = prices.isnull().sum()
    if null_count > len(prices) * 0.05:  # >5% nulls
        report["issues"].append(f"Too many nulls: {null_count}")
        logger.error(f"{name} has {null_count} null values")
        return False, report
    elif null_count > 0:
        report["warnings"].append(f"Found {null_count} nulls")
    
    # Check for zeros
    zero_count = (prices == 0).sum()
    if zero_count > len(prices) * 0.05:  # >5% zeros
        report["issues"].append(f"Too many zeros: {zero_count}")
        logger.error(f"{name} has {zero_count} zero values")
        return False, report
    
    # Check for negative prices
    if (prices < 0).any():
        neg_count = (prices < 0).sum()
        report["issues"].append(f"Found negative prices: {neg_count}")
        logger.error(f"{name} has {neg_count} negative values")
        return False, report
    
    # Check for extreme values
    q1 = prices.quantile(0.01)
    q99 = prices.quantile(0.99)
    outliers = ((prices < q1) | (prices > q99)).sum()
    pct_outliers = (outliers / len(prices)) * 100
    if pct_outliers > 5:
        report["warnings"].append(f"High outliers: {pct_outliers:.1f}%")
        logger.warning(f"{name} has {pct_outliers:.1f}% outliers")
    
    # Check volatility
    returns = prices.pct_change().dropna()
    volatility = returns.std()
    mean_return = returns.mean()
    if volatility > 0.5:  # >50% daily volatility is extreme
        report["warnings"].append(f"High volatility: {volatility:.1%}")
        logger.warning(f"{name} has high volatility: {volatility:.1%}")
    
    logger.info(f"{name} validation passed. Length: {len(prices)}, Volatility: {volatility:.2%}")
    return True, report


def validate_features(
    X: np.ndarray,
    name: str = "Features",
) -> Tuple[bool, Dict]:
    """
    Validate feature matrix.
    
    Args:
        X: Feature matrix (2D or 3D)
        name: Name for logging
    
    Returns:
        (is_valid, report_dict)
    """
    report = {
        "name": name,
        "shape": X.shape,
        "issues": [],
        "warnings": [],
    }
    
    # Check for NaN
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        report["issues"].append(f"Found NaN values: {nan_count}")
        logger.error(f"{name} contains {nan_count} NaN values")
        return False, report
    
    # Check for Inf
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        report["issues"].append(f"Found Inf values: {inf_count}")
        logger.error(f"{name} contains {inf_count} Inf values")
        return False, report
    
    # Check shape
    if len(X.shape) == 3 and X.shape[0] < 10:
        report["warnings"].append(f"Small sample size: {X.shape[0]}")
        logger.warning(f"{name} has small sample: {X.shape[0]}")
    
    logger.info(f"{name} validation passed. Shape: {X.shape}")
    return True, report


def check_data_quality(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Comprehensive data quality check.
    
    Args:
        df: Input dataframe
        verbose: Print detailed report
    
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("Starting comprehensive data quality check")
    
    # Check if empty
    if df.empty:
        logger.error("DataFrame is empty!")
        return False
    
    # Check duplicates
    dup_ratio = df.duplicated().sum() / len(df)
    if dup_ratio > 0.1:
        logger.error(f"High duplicate ratio: {dup_ratio:.1%}")
        return False
    
    # Check nulls
    null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if null_ratio > 0.05:
        logger.error(f"High null ratio: {null_ratio:.1%}")
        return False
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            logger.warning(f"Column '{col}' has zero variance")
        if (df[col] < 0).any():
            logger.warning(f"Column '{col}' has negative values")
    
    logger.info("Data quality check passed!")
    if verbose:
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Duplicates: {df.duplicated().sum()}")
        print(f"✓ Nulls: {df.isnull().sum().sum()}")
        print(f"✓ Numeric columns: {len(numeric_cols)}")
    
    return True
