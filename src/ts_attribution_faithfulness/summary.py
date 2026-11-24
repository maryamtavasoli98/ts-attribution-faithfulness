"""Aggregate raw per-seed records into the numbers shown in the README and figures."""

from __future__ import annotations

import math

import numpy as np


def mean_std(v: list[float]) -> tuple[float, float]:
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s


def aggregate_faithfulness(records: list[dict]) -> dict:
    """{(arch, method, baseline): {metric: (mean, std over seeds)}}"""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["arch"], r["method"], r["eval_baseline"]), []).append(r)
    return {
        k: {m: mean_std([r[m] for r in v]) for m in ("deletion_auc", "insertion_auc", "gap")} for k, v in groups.items()
    }


def paired_difference(records: list[dict], arch: str, method: str, ref: str, baseline: str, n_boot: int = 2000):
    """Mean per-window gap difference method - ref (windows and seeds pooled) and a 95 % bootstrap interval."""

    def pooled(m):
        rs = sorted(
            (r for r in records if r["arch"] == arch and r["method"] == m and r["eval_baseline"] == baseline),
            key=lambda r: r["seed"],
        )
        return np.concatenate([np.asarray(r["gap_per_window"]) for r in rs])

    d = pooled(method) - pooled(ref)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_vs_random(records: list[dict], arch: str, method: str, baseline: str) -> float:
    """Share of (seed, window) pairs where the method's gap beats the random attribution's gap on the same window."""
    sel = {
        m: {
            r["seed"]: r["gap_per_window"]
            for r in records
            if r["arch"] == arch and r["method"] == m and r["eval_baseline"] == baseline
        }
        for m in (method, "random")
    }
    wins = tot = 0
    for seed, g in sel[method].items():
        for a, b in zip(g, sel["random"][seed], strict=True):
            wins += a > b
            tot += 1
    return wins / tot
