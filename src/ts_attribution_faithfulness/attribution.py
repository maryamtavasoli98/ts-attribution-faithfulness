"""Attribution methods for a scalar forecast, all at the level of feature *groups*.

A group is a (channel, block of consecutive lags) cell of the input window. The explained scalar is
``f(x) = mean over the horizon of the forecast of the target channel``. Every method returns a signed
attribution of shape (N, G) relative to a baseline window (here the training mean, i.e. zeros after
standardisation). Implemented from scratch: integrated gradients, gradient x input, group occlusion,
permutation-sampling Shapley values and a LIME-style weighted ridge surrogate.
"""

from __future__ import annotations

import math

import torch


def group_index(lookback: int, channels: int, block: int) -> tuple[torch.Tensor, int]:
    """Map each (lag, channel) cell to a group id; returns (gid of shape (L, C), number of groups)."""
    t = torch.arange(lookback)[:, None] // block
    c = torch.arange(channels)[None, :]
    nb = math.ceil(lookback / block)
    return (t * channels + c).long(), nb * channels


def aggregate(attr: torch.Tensor, gid: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Sum element attributions (N, L, C) into group attributions (N, G)."""
    out = torch.zeros(attr.shape[0], n_groups, dtype=attr.dtype, device=attr.device)
    return out.index_add_(1, gid.reshape(-1).to(attr.device), attr.reshape(attr.shape[0], -1))


@torch.no_grad()
def scalar(model, x: torch.Tensor) -> torch.Tensor:
    """Explained quantity: mean forecast over the horizon, shape (N,)."""
    return model(x).mean(-1)


@torch.no_grad()
def eval_masks(model, x, base, masks, gid, max_batch: int = 16384) -> torch.Tensor:
    """Forecast scalar for masked windows. masks (N, M, G) in {0,1}: 1 keeps the feature, 0 uses the baseline."""
    n, m, _ = masks.shape
    out = torch.empty(n, m, device=x.device)
    step = max(1, max_batch // m)
    gid = gid.to(x.device)
    for i in range(0, n, step):
        xs, bs, ms = x[i : i + step], base[i : i + step], masks[i : i + step].to(x.dtype)
        el = ms[..., gid]  # (n, M, L, C)
        xm = el * xs[:, None] + (1 - el) * bs[:, None]
        out[i : i + step] = scalar(model, xm.reshape(-1, *x.shape[1:])).reshape(len(xs), m)
    return out


def grad_x_input(model, x, base, gid, n_groups) -> torch.Tensor:
    xr = x.clone().requires_grad_(True)
    scalar_out = model(xr).mean(-1).sum()
    (g,) = torch.autograd.grad(scalar_out, xr)
    return aggregate(g * (x - base), gid, n_groups)


def integrated_gradients(model, x, base, gid, n_groups, steps: int = 32) -> torch.Tensor:
    """Riemann (midpoint) integrated gradients along the straight path baseline -> x."""
    total = torch.zeros_like(x)
    for k in range(steps):
        a = (k + 0.5) / steps
        xi = (base + a * (x - base)).requires_grad_(True)
        (g,) = torch.autograd.grad(model(xi).mean(-1).sum(), xi)
        total += g
    return aggregate(total / steps * (x - base), gid, n_groups)


def occlusion(model, x, base, gid, n_groups) -> torch.Tensor:
    """f(x) - f(x with one group set to baseline)."""
    n = x.shape[0]
    masks = 1 - torch.eye(n_groups, device=x.device)[None].expand(n, -1, -1)
    full = scalar(model, x)
    return full[:, None] - eval_masks(model, x, base, masks, gid)


def shapley_sampling(model, x, base, gid, n_groups, perms: int = 20, seed: int = 0, chunk: int = 8) -> torch.Tensor:
    """Permutation-sampling Shapley values of the group game v(S) = f(x with only S kept)."""
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    phi = torch.zeros(n, n_groups, device=x.device)
    steps = torch.arange(n_groups + 1)
    for i in range(0, n, chunk):
        nb = min(chunk, n - i)
        order = torch.rand(nb, perms, n_groups, generator=g_).argsort(-1)  # (nb, P, G)
        rank = order.argsort(-1)  # position of each group in its permutation
        masks = (rank[:, :, None, :] < steps[None, None, :, None]).float()  # (nb, P, G+1, G)
        v = eval_masks(model, x[i : i + nb], base[i : i + nb], masks.reshape(nb, -1, n_groups).to(x.device), gid)
        v = v.reshape(nb, perms, n_groups + 1)
        d = v[..., 1:] - v[..., :-1]  # contribution of the j-th group in each permutation
        phi[i : i + nb] = d.gather(-1, rank.to(x.device)).mean(1)
    return phi


def lime_style(model, x, base, gid, n_groups, samples: int = 512, ridge: float = 1e-2, seed: int = 0) -> torch.Tensor:
    """Weighted ridge regression of the forecast on random group on/off masks (kernel on the share removed)."""
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    p = torch.rand(n, samples, 1, generator=g_) * 0.8 + 0.1
    z = (torch.rand(n, samples, n_groups, generator=g_) < p).float()
    v = eval_masks(model, x, base, z.to(x.device), gid).cpu().double()
    removed = 1 - z.mean(-1).double()
    w = torch.exp(-(removed**2) / 0.5**2)
    zz = torch.cat([z.double(), torch.ones(n, samples, 1, dtype=torch.double)], -1)
    a = zz.transpose(1, 2) @ (w[..., None] * zz) + ridge * torch.eye(n_groups + 1, dtype=torch.double)
    b = zz.transpose(1, 2) @ (w * v)[..., None]
    coef = torch.linalg.solve(a, b)[..., 0]
    return coef[:, :n_groups].float().to(x.device)


def random_attribution(n: int, n_groups: int, seed: int = 0, device="cpu") -> torch.Tensor:
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, n_groups, generator=g_).to(device)


METHODS = ["random", "grad_x_input", "integrated_gradients", "occlusion", "shapley_sampling", "lime_style"]


def explain(method: str, model, x, base, gid, n_groups, seed: int = 0) -> torch.Tensor:
    if method == "random":
        return random_attribution(x.shape[0], n_groups, seed, x.device)
    if method == "grad_x_input":
        return grad_x_input(model, x, base, gid, n_groups)
    if method == "integrated_gradients":
        return integrated_gradients(model, x, base, gid, n_groups)
    if method == "occlusion":
        return occlusion(model, x, base, gid, n_groups)
    if method == "shapley_sampling":
        return shapley_sampling(model, x, base, gid, n_groups, seed=seed)
    if method == "lime_style":
        return lime_style(model, x, base, gid, n_groups, seed=seed)
    raise KeyError(method)


def linear_exact(model, x, base, gid, n_groups) -> torch.Tensor:
    """Exact group attribution for the linear forecaster: mean-over-horizon weight x (x - baseline)."""
    w = model.fc.weight.detach().mean(0).reshape(x.shape[1:])
    return aggregate(w[None] * (x - base), gid, n_groups)
