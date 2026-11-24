#!/usr/bin/env python3
"""Figures from results/metrics/attribution_results.json."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ts_attribution_faithfulness.common import FIGURES, METRICS
from ts_attribution_faithfulness.summary import aggregate_faithfulness

COL = {
    "random": "#999999",
    "grad_x_input": "#d95f02",
    "integrated_gradients": "#7570b3",
    "occlusion": "#1b9e77",
    "shapley_sampling": "#e7298a",
    "lime_style": "#66a61e",
    "exact_linear": "#000000",
}


def curves_fig(recs, archs, base, fname, note):
    fig, axes = plt.subplots(2, len(archs), figsize=(4.2 * len(archs), 6), sharex=True)
    for j, arch in enumerate(archs):
        for i, key in enumerate(("del_curve", "ins_curve")):
            ax = axes[i, j]
            for m in COL:
                rs = [r for r in recs if r["arch"] == arch and r["method"] == m and r["eval_baseline"] == base]
                if not rs:
                    continue
                y = np.mean([r[key] for r in rs], 0)
                ax.plot(
                    np.linspace(0, 1, len(y)), y, color=COL[m], label=m, lw=1.6, ls="--" if m == "exact_linear" else "-"
                )
            ax.set_title(f"{arch}: {'deletion' if i == 0 else 'insertion'}{note}", fontsize=9)
            ax.set_ylabel("|forecast change| (std. units)")
            if i == 1:
                ax.set_xlabel("fraction of groups replaced / restored")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = json.loads((METRICS / "attribution_results.json").read_text())
    recs = res["faithfulness"]
    archs = res["config"]["archs"]
    G = res["n_groups"]

    # 1. mean deletion / insertion curves, one column per architecture (all groups, then dominant group pinned)
    curves_fig(recs, archs, "mean", "curves.png", "")
    curves_fig(recs, archs, "mean_pin1", "curves_pin1.png", " (dominant group pinned)")

    # 2. gap bars with seed std, both evaluation baselines
    agg = aggregate_faithfulness(recs)
    methods = [m for m in COL if m != "exact_linear"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, base in zip(axes, ("mean", "local_mean", "mean_pin1"), strict=True):
        w = 0.8 / len(archs)
        for k, arch in enumerate(archs):
            mu = [agg[(arch, m, base)]["gap"][0] for m in methods]
            sd = [agg[(arch, m, base)]["gap"][1] for m in methods]
            ax.bar(np.arange(len(methods)) + k * w, mu, w, yerr=sd, label=arch, capsize=2)
        ax.set_xticks(np.arange(len(methods)) + w, [m.replace("_", "\n") for m in methods], fontsize=7)
        ax.set_title(
            f"deletion AUC - insertion AUC (higher = more faithful)\nperturbation baseline: {base.replace('_', '-')}",
            fontsize=8,
        )
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "faithfulness_gap.png", dpi=140)
    plt.close(fig)

    # 3. where do the methods look? mean |attribution| share per (lag block, channel)
    chans = res["groups"]["channels"]
    nb = G // len(chans)
    show = ["integrated_gradients", "occlusion", "shapley_sampling", "lime_style"]
    fig, axes = plt.subplots(
        len(archs), len(show), figsize=(3.3 * len(show), 2.2 * len(archs)), sharex=True, sharey=True
    )
    for i, arch in enumerate(archs):
        for j, m in enumerate(show):
            prof = np.array(res["lag_profiles"][f"{arch}/{m}"]).reshape(nb, len(chans))
            axes[i, j].imshow(prof.T, aspect="auto", cmap="viridis")
            axes[i, j].set_title(f"{arch} / {m}", fontsize=7)
            axes[i, j].set_yticks(range(len(chans)), chans, fontsize=6)
            axes[i, j].set_xticks(
                range(nb), [f"-{res['groups']['lookback'] - b * res['groups']['block']}" for b in range(nb)], fontsize=5
            )
    fig.supxlabel("lag block (hours before forecast origin, start of block)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "attribution_maps.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
