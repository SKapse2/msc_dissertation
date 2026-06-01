"""Preprocessing pipeline for NAB time-series anomaly detection.

This module loads a NAB CSV, splits it temporally into train/val/test
(excluding labelled anomaly windows from training), normalises with
z-score statistics from training data only, and windows the series
into fixed-length overlapping sequences for the autoencoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --- Configuration ---------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    """Temporal split boundaries (inclusive of start, exclusive of end)."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# Default split for machine_temperature_system_failure.csv
DEFAULT_SPLIT = SplitConfig(
    train_start=pd.Timestamp("2013-12-02"),
    train_end=pd.Timestamp("2014-01-25"),
    val_start=pd.Timestamp("2014-01-25"),
    val_end=pd.Timestamp("2014-01-27"),
    test_start=pd.Timestamp("2014-01-27"),
    test_end=pd.Timestamp("2014-02-19 23:59:59"),
)

WINDOW_SIZE = 60          # 60 readings = 5 hours at 5-min sampling
WINDOW_STRIDE = 1         # one window per timestep; high overlap


# --- Data loading ----------------------------------------------------------

def load_nab_stream(csv_path: Path) -> pd.DataFrame:
    """Load a NAB CSV with parsed timestamps."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_anomaly_windows(
    labels_file: Path, target_file: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return [(start, end), ...] anomaly windows for the given file."""
    with open(labels_file) as f:
        all_labels = json.load(f)
    if target_file not in all_labels:
        raise KeyError(f"{target_file} not found in labels file")
    return [
        (pd.to_datetime(start), pd.to_datetime(end))
        for start, end in all_labels[target_file]
    ]


# --- Splitting -------------------------------------------------------------

def split_by_time(
    df: pd.DataFrame, split: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test by timestamp."""
    train = df[(df.timestamp >= split.train_start) & (df.timestamp < split.train_end)]
    val   = df[(df.timestamp >= split.val_start)   & (df.timestamp < split.val_end)]
    test  = df[(df.timestamp >= split.test_start)  & (df.timestamp < split.test_end)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def remove_anomaly_windows(
    df: pd.DataFrame,
    anomaly_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Drop rows that fall inside any labelled anomaly window.

    Used on the training set only. The autoencoder must learn 'normal'.
    """
    mask = pd.Series(True, index=df.index)
    for start, end in anomaly_windows:
        in_window = (df.timestamp >= start) & (df.timestamp <= end)
        mask &= ~in_window
    return df[mask].reset_index(drop=True)


# --- Normalisation ---------------------------------------------------------

@dataclass(frozen=True)
class Scaler:
    """Z-score scaler fitted on training data."""
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def fit_scaler(values: np.ndarray) -> Scaler:
    """Compute mean and std for z-score normalisation."""
    return Scaler(mean=float(values.mean()), std=float(values.std()))


# --- Windowing -------------------------------------------------------------

def window_series(
    values: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE
) -> np.ndarray:
    """Slice a 1D series into overlapping windows.

    Returns an array of shape (num_windows, window_size, 1) — the trailing
    1 is the channel dimension expected by Conv1D layers.
    """
    if values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {values.shape}")
    n = len(values)
    if n < window_size:
        raise ValueError(f"Series of length {n} shorter than window {window_size}")
    starts = np.arange(0, n - window_size + 1, stride)
    windows = np.stack([values[s:s + window_size] for s in starts])
    return windows[..., np.newaxis]  # (num_windows, window_size, 1)

def window_dataframe_by_segments(
    df: pd.DataFrame,
    scaler: Scaler,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    expected_interval: pd.Timedelta = pd.Timedelta(minutes=5),
) -> np.ndarray:
    """Window a dataframe respecting timestamp gaps.

    Splits the dataframe into continuous segments wherever there's a
    timestamp gap larger than expected_interval, then windows each
    segment independently. Avoids windows that span across removed
    anomaly periods.
    """
    # Identify segment boundaries: gaps larger than 1.5x expected interval
    gaps = df['timestamp'].diff() > expected_interval * 1.5
    segment_id = gaps.cumsum()

    all_windows = []
    for _, segment in df.groupby(segment_id):
        if len(segment) < window_size:
            continue  # segment too short to window
        values = scaler.transform(segment['value'].values)
        windows = window_series(values, window_size, stride)
        all_windows.append(windows)

    if not all_windows:
        raise ValueError("No segments long enough to window")
    return np.concatenate(all_windows, axis=0)