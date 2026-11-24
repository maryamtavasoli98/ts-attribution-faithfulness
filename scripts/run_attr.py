#!/usr/bin/env python3
"""Attributions + deletion/insertion faithfulness + sanity + stability for every trained forecaster."""

import time

import torch

from ts_attribution_faithfulness.attribution import METHODS, explain, group_index, linear_exact
from ts_attribution_faithfulness.common import CSV, RUNS, config, device, environment, write_json
from ts_attribution_faithfulness.data import load_splits, make_windows
from ts_attribution_faithfulness.faithfulness import curves, scores, spearman, topk_jaccard
from ts_attribution_faithfulness.models import build


def main() -> None:
    cfg, dev = config(), device()
    s = load_splits(CSV, cfg["lookback"])
    L, H, C = cfg["lookback"], cfg["horizon"], s.test.shape[1]
    xt, _ = make_windows(s.test, L, H, s.target_idx)
    sel = torch.linspace(0, len(xt) - 1, cfg["n_windows"]).round().long()
    x = torch.from_numpy(xt)[sel].to(dev)
    gid, G = group_index(L, C, cfg["block"])
    bases = {"mean": torch.zeros_like(x), "local_mean": x.mean(1, keepdim=True).expand_as(x).contiguous()}
    base_attr = bases["mean"]
    res = {
        "environment": environment(),
        "config": cfg,
        "n_groups": G,
        "groups": {"lookback": L, "channels": s.columns, "block": cfg["block"]},
        "faithfulness": [],
        "sanity_linear": [],
        "stability": [],
        "agreement": [],
        "pinned_group": [],
        "lag_profiles": {},
        "timing_seconds": {},
    }
    t_start = time.time()
    for arch in cfg["archs"]:
        attrs: dict[str, list[torch.Tensor]] = {}
        for seed in cfg["seeds"]:
            model = build(arch, L, C, H).to(dev)
            model.load_state_dict(torch.load(RUNS / f"{arch}_s{seed}.pt", map_location=dev))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            methods = ["shapley_sampling"] + [m for m in METHODS if m != "shapley_sampling"]
            methods += ["exact_linear"] if arch == "linear" else []
            pin = None
            for m in methods:
                t0 = time.time()
                a = (
                    linear_exact(model, x, base_attr, gid, G)
                    if m == "exact_linear"
                    else explain(m, model, x, base_attr, gid, G, seed=seed)
                )
                a = a.detach()
                if dev.type == "mps":
                    torch.mps.synchronize()
                dt = time.time() - t0
                res["timing_seconds"].setdefault(f"{arch}/{m}", []).append(round(dt, 2))
                attrs.setdefault(m, []).append(a.cpu())
                if m == "shapley_sampling":
                    pin = int(a.abs().mean(0).argmax())  # dominant group of this model, shared by all methods
                    res["pinned_group"].append(
                        {"arch": arch, "seed": seed, "group": pin, "lag_block": pin // C, "channel": s.columns[pin % C]}
                    )
                evals = [
                    ("mean", bases["mean"], None),
                    ("local_mean", bases["local_mean"], None),
                    ("mean_pin1", bases["mean"], pin),
                ]
                for bname, b, pn in evals:
                    c = curves(model, x, b, a, gid, pn)
                    sc = scores(c)
                    res["faithfulness"].append(
                        {
                            "arch": arch,
                            "seed": seed,
                            "method": m,
                            "eval_baseline": bname,
                            **{k: float(v.mean()) for k, v in sc.items()},
                            "gap_per_window": [round(float(v), 5) for v in sc["gap"].cpu()],
                            "del_curve": [round(float(v), 5) for v in c["del"].mean(0).cpu()],
                            "ins_curve": [round(float(v), 5) for v in c["ins"].mean(0).cpu()],
                        }
                    )
                if arch == "linear" and m not in ("exact_linear", "random"):
                    exact = linear_exact(model, x, base_attr, gid, G)
                    res["sanity_linear"].append(
                        {
                            "seed": seed,
                            "method": m,
                            "spearman_abs_vs_exact": float(spearman(a.abs(), exact.abs()).mean()),
                            "top10_jaccard_vs_exact": float(topk_jaccard(a, exact, cfg["topk"]).mean()),
                            "max_abs_err_vs_exact": float((a - exact).abs().max()),
                        }
                    )
                print(arch, seed, m, f"{dt:.1f}s", flush=True)
        ref = attrs["shapley_sampling"]
        for m, lst in attrs.items():
            for i, a in enumerate(lst):
                res["agreement"].append(
                    {
                        "arch": arch,
                        "seed": cfg["seeds"][i],
                        "method": m,
                        "spearman_abs_vs_shapley": float(spearman(a.abs(), ref[i].abs()).mean()),
                        "top10_jaccard_vs_shapley": float(topk_jaccard(a, ref[i], cfg["topk"]).mean()),
                    }
                )
        for m, lst in attrs.items():
            js = []
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    js.append(float(topk_jaccard(lst[i], lst[j], cfg["topk"]).mean()))
            res["stability"].append({"arch": arch, "method": m, "top10_jaccard_across_seeds": sum(js) / len(js)})
            prof = torch.stack([a.abs() / a.abs().sum(-1, keepdim=True) for a in lst]).mean((0, 1))
            res["lag_profiles"][f"{arch}/{m}"] = [round(float(v), 6) for v in prof]
    res["total_seconds"] = round(time.time() - t_start, 1)
    write_json("attribution_results.json", res)
    print("done", res["total_seconds"])


if __name__ == "__main__":
    main()
