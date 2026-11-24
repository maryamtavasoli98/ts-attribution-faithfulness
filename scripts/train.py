#!/usr/bin/env python3
"""Train linear / LSTM / TCN forecasters on ETTh1 for each seed; save weights (git-ignored) + accuracy JSON."""

import time

import numpy as np
import torch
from torch import nn

from ts_attribution_faithfulness.common import CSV, RUNS, config, device, environment, write_json
from ts_attribution_faithfulness.data import load_splits, make_windows
from ts_attribution_faithfulness.models import build


def batches(x, y, bs, shuffle, gen):
    idx = torch.randperm(len(x), generator=gen) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), bs):
        j = idx[i : i + bs]
        yield x[j], y[j]


@torch.no_grad()
def predict(model, x, dev, bs=1024):
    model.eval()
    return torch.cat([model(x[i : i + bs].to(dev)).cpu() for i in range(0, len(x), bs)])


def main() -> None:
    cfg, dev = config(), device()
    s = load_splits(CSV, cfg["lookback"])
    L, H, C = cfg["lookback"], cfg["horizon"], s.train.shape[1]
    data = {
        k: tuple(map(torch.from_numpy, make_windows(getattr(s, k), L, H, s.target_idx)))
        for k in ("train", "val", "test")
    }
    sd = float(s.std[s.target_idx])
    out = {"environment": environment(), "config": cfg, "target_std_celsius": sd, "runs": []}
    xt, yt = data["test"]
    naive = xt[:, -1, s.target_idx][:, None].expand(-1, H) - yt
    out["naive_last_value"] = {"mse": float((naive**2).mean()), "mae": float(naive.abs().mean())}
    RUNS.mkdir(exist_ok=True)
    for arch in cfg["archs"]:
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = build(arch, L, C, H).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            best, best_state, bad, t0 = float("inf"), None, 0, time.time()
            for epoch in range(cfg["epochs"]):
                model.train()
                for xb, yb in batches(*data["train"], cfg["batch"], True, gen):
                    opt.zero_grad()
                    nn.functional.mse_loss(model(xb.to(dev)), yb.to(dev)).backward()
                    opt.step()
                val = float(((predict(model, data["val"][0], dev) - data["val"][1]) ** 2).mean())
                if val < best - 1e-5:
                    best, bad = val, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            model.load_state_dict(best_state)
            err = predict(model, xt, dev) - yt
            rec = {
                "arch": arch,
                "seed": seed,
                "epochs_run": epoch + 1,
                "val_mse": best,
                "test_mse": float((err**2).mean()),
                "test_mae": float(err.abs().mean()),
                "test_mae_celsius": float(err.abs().mean()) * sd,
                "params": sum(p.numel() for p in model.parameters()),
                "train_seconds": round(time.time() - t0, 1),
            }
            torch.save(best_state, RUNS / f"{arch}_s{seed}.pt")
            out["runs"].append(rec)
            print(rec, flush=True)
    write_json("forecast_accuracy.json", out)


if __name__ == "__main__":
    main()
