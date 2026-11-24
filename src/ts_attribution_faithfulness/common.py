"""Shared helpers for the scripts: paths, config, device, environment record."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
RUNS = ROOT / "runs"
CSV = ROOT / "data" / "raw" / "ETTh1.csv"


def config() -> dict:
    return json.loads((ROOT / "configs" / "default.json").read_text())


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device()),
        "machine": platform.machine(),
        "platform": platform.system(),
    }


def write_json(name: str, obj) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / name).write_text(json.dumps(obj, indent=1))
