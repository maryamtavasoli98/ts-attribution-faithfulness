"""Deletion / insertion faithfulness and explanation-agreement metrics."""

from __future__ import annotations

import torch

from .attribution import eval_masks, scalar


def _trapz(y: torch.Tensor) -> torch.Tensor:
    """Area under a curve sampled at k/G, k = 0..G (x axis normalised to [0, 1]); y shape (N, G+1)."""
    return ((y[:, 1:] + y[:, :-1]) / 2).mean(-1)


def rank_masks(attr: torch.Tensor, mode: str, pin: int | None = None) -> torch.Tensor:
    """Masks (N, K+1, G) for deletion (mode='del') or insertion ('ins'); most important group (by |attr|) first.

    With ``pin`` (a group id) that group is never deleted and always present when inserting; the ranking then
    covers the remaining G-1 groups only (K = G-1), which tests the tail of the explanation.
    """
    n, g = attr.shape
    key = -attr.abs()
    if pin is not None:
        key = key.clone()
        key[:, pin] = float("inf")  # sorts last
    rank = key.argsort(-1).argsort(-1)
    steps = g if pin is None else g - 1
    k = torch.arange(steps + 1, device=attr.device)
    top_k = rank[:, None, :] < k[None, :, None]  # first k groups selected
    if pin is not None:
        top_k[:, :, pin] = False
    m = (~top_k).float() if mode == "del" else top_k.float()
    if pin is not None:
        m[:, :, pin] = 1.0
    return m


def curves(model, x, base, attr, gid, pin: int | None = None) -> dict[str, torch.Tensor]:
    """Mean-forecast deviation |f(x_k) - f(x)| after deleting / inserting the k most important groups."""
    f0 = scalar(model, x)
    out = {}
    for mode in ("del", "ins"):
        v = eval_masks(model, x, base, rank_masks(attr, mode, pin), gid)
        out[mode] = (v - f0[:, None]).abs()
    return out


def scores(c: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-window AUCs: deletion (higher = more faithful), insertion (lower = more faithful), and their gap."""
    d, i = _trapz(c["del"]), _trapz(c["ins"])
    return {"deletion_auc": d, "insertion_auc": i, "gap": d - i}


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of two (N, G) tensors (no tie correction; inputs are continuous)."""
    ra, rb = a.argsort(-1).argsort(-1).float(), b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1) + 1e-12)


def topk_jaccard(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """Row-wise Jaccard overlap of the top-k groups by |attribution|."""
    ta = (-a.abs()).argsort(-1)[:, :k]
    tb = (-b.abs()).argsort(-1)[:, :k]
    ma = torch.zeros_like(a, dtype=torch.bool).scatter_(1, ta, True)
    mb = torch.zeros_like(b, dtype=torch.bool).scatter_(1, tb, True)
    inter = (ma & mb).sum(-1).float()
    return inter / (ma | mb).sum(-1).float()


def rank_correlation(a: list[float], b: list[float]) -> float:
    """Spearman correlation of two score lists over the same items."""
    ta = torch.tensor(a)
    tb = torch.tensor(b)
    return float(spearman(ta[None], tb[None])[0])
