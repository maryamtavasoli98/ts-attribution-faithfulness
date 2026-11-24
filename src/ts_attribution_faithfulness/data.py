"""ETTh1 loading, chronological split and sliding windows.

ETTh1 (Zhou et al., 2021) is hourly transformer data with 7 channels; the last one (OT, oil temperature)
is the forecast target. The split follows the common 12 / 4 / 4 month protocol (8640 / 2880 / 2880 rows).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
TARGET = "OT"
N_TRAIN, N_VAL, N_TEST = 8640, 2880, 2880


def download(dest: Path) -> Path:
    """Fetch ETTh1.csv (2.6 MB) into ``dest`` unless it is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(URL, dest)  # noqa: S310 - fixed https URL
    return dest


@dataclass
class Splits:
    """Standardised arrays (train statistics only) and metadata."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    target_idx: int


def load_splits(csv: Path, lookback: int) -> Splits:
    """Return standardised train/val/test arrays. Val/test include ``lookback`` rows of history."""
    df = pd.read_csv(csv)
    cols = [c for c in df.columns if c != "date"]
    arr = df[cols].to_numpy(dtype=np.float32)
    mean = arr[:N_TRAIN].mean(0)
    std = arr[:N_TRAIN].std(0) + 1e-8
    z = (arr - mean) / std
    train = z[:N_TRAIN]
    val = z[N_TRAIN - lookback : N_TRAIN + N_VAL]
    test = z[N_TRAIN + N_VAL - lookback : N_TRAIN + N_VAL + N_TEST]
    return Splits(train, val, test, cols, mean, std, cols.index(TARGET))


def make_windows(series: np.ndarray, lookback: int, horizon: int, target_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows with stride 1: x (N, lookback, C), y (N, horizon) for the target channel."""
    n = len(series) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("series too short for the requested window")
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    x = series[idx]
    yidx = lookback + np.arange(horizon)[None, :] + np.arange(n)[:, None]
    y = series[yidx, target_idx]
    return x.astype(np.float32), y.astype(np.float32)
