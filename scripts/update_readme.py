#!/usr/bin/env python3
"""Fill the ``<!-- table:NAME -->`` blocks of README.md from results/metrics/*.json (nothing is typed by hand)."""

import json
import re

from ts_attribution_faithfulness.common import METRICS, ROOT
from ts_attribution_faithfulness.summary import aggregate_faithfulness, mean_std, paired_difference, paired_vs_random

TARGETS = [ROOT / "README.md", ROOT / "report" / "report.md"]


def pm(t, d=3):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def load(name):
    return json.loads((METRICS / name).read_text())


def table_accuracy():
    a = load("forecast_accuracy.json")
    rows = ["| model | params | test MAE (°C) | test MSE (std. units) | epochs (early stop) |", "|---|---|---|---|---|"]
    for arch in a["config"]["archs"]:
        rs = [r for r in a["runs"] if r["arch"] == arch]
        rows.append(
            f"| {arch} | {rs[0]['params']:,} | {pm(mean_std([r['test_mae_celsius'] for r in rs]), 2)} | "
            f"{pm(mean_std([r['test_mse'] for r in rs]), 3)} | {pm(mean_std([float(r['epochs_run']) for r in rs]), 1)} |"
        )
    n = a["naive_last_value"]
    rows.append(f"| naive: repeat last OT value | 0 | {n['mae'] * a['target_std_celsius']:.2f} | {n['mse']:.3f} | – |")
    return "\n".join(rows)


def table_main():
    r = load("attribution_results.json")
    recs = r["faithfulness"]
    agg = aggregate_faithfulness(recs)
    stab = {(s["arch"], s["method"]): s["top10_jaccard_across_seeds"] for s in r["stability"]}
    rows = [
        "| model | attribution | deletion AUC ↑ | insertion AUC ↓ | gap ↑ (mean baseline) | gap ↑ (local-mean baseline) | gap ↑ (dominant group pinned) | beats random (windows) | top-10 stability across seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arch in r["config"]["archs"]:
        order = [
            "random",
            "grad_x_input",
            "integrated_gradients",
            "occlusion",
            "shapley_sampling",
            "lime_style",
            "exact_linear",
        ]
        for m in [m for m in order if (arch, m, "mean") in agg]:
            a = agg[(arch, m, "mean")]
            win = "–" if m == "random" else f"{100 * paired_vs_random(recs, arch, m, 'mean'):.0f} %"
            rows.append(
                f"| {arch} | {m} | {pm(a['deletion_auc'])} | {pm(a['insertion_auc'])} | {pm(a['gap'])} | "
                f"{pm(agg[(arch, m, 'local_mean')]['gap'])} | {pm(agg[(arch, m, 'mean_pin1')]['gap'])} | {win} | {stab[(arch, m)]:.2f} |"
            )
    return "\n".join(rows)


def table_sanity():
    r = load("attribution_results.json")
    rows = [
        "| attribution | Spearman (absolute attributions) vs exact | top-10 overlap vs exact | max abs error vs exact |",
        "|---|---|---|---|",
    ]
    for m in sorted({s["method"] for s in r["sanity_linear"]}):
        ss = [s for s in r["sanity_linear"] if s["method"] == m]
        rows.append(
            f"| {m} | {sum(s['spearman_abs_vs_exact'] for s in ss) / len(ss):.3f} | "
            f"{sum(s['top10_jaccard_vs_exact'] for s in ss) / len(ss):.3f} | {max(s['max_abs_err_vs_exact'] for s in ss):.2g} |"
        )
    return "\n".join(rows)


def table_agreement():
    r = load("attribution_results.json")
    recs = r["faithfulness"]
    rows = [
        "| model | attribution | Spearman vs Shapley | top-10 overlap | gap diff. vs Shapley [95 % CI] | gap diff., group pinned [95 % CI] |",
        "|---|---|---|---|---|---|",
    ]
    for arch in r["config"]["archs"]:
        for m in ("grad_x_input", "integrated_gradients", "occlusion", "lime_style", "random"):
            ag = [x for x in r["agreement"] if x["arch"] == arch and x["method"] == m]
            sp = sum(x["spearman_abs_vs_shapley"] for x in ag) / len(ag)
            jc = sum(x["top10_jaccard_vs_shapley"] for x in ag) / len(ag)
            d, lo, hi = paired_difference(recs, arch, m, "shapley_sampling", "mean")
            d2, lo2, hi2 = paired_difference(recs, arch, m, "shapley_sampling", "mean_pin1")
            rows.append(
                f"| {arch} | {m} | {sp:.3f} | {jc:.3f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {d2:+.4f} [{lo2:+.4f}, {hi2:+.4f}] |"
            )
    return "\n".join(rows)


def table_early():
    r = load("attribution_results.json")
    k = round(0.1 * r["n_groups"])
    rows = [
        f"| model | attribution | forecast change after removing the top {k} of {r['n_groups']} groups (10 %) ↑ | after keeping only the top {k} groups ↓ |",
        "|---|---|---|---|",
    ]
    for arch in r["config"]["archs"]:
        for m in ("random", "grad_x_input", "integrated_gradients", "occlusion", "shapley_sampling", "lime_style"):
            rs = [
                x for x in r["faithfulness"] if x["arch"] == arch and x["method"] == m and x["eval_baseline"] == "mean"
            ]
            d = mean_std([x["del_curve"][k] for x in rs])
            i = mean_std([x["ins_curve"][k] for x in rs])
            rows.append(f"| {arch} | {m} | {pm(d)} | {pm(i)} |")
    return "\n".join(rows)


def table_pinned():
    r = load("attribution_results.json")
    blk, look = r["groups"]["block"], r["groups"]["lookback"]
    rows = ["| model | seed | dominant group (channel, lags before the forecast origin) |", "|---|---|---|"]
    for p in r["pinned_group"]:
        lo = look - p["lag_block"] * blk
        rows.append(f"| {p['arch']} | {p['seed']} | {p['channel']}, hours -{lo} to -{lo - blk + 1} |")
    return "\n".join(rows)


def table_timing():
    r = load("attribution_results.json")
    rows = ["| attribution | linear | lstm | tcn |", "|---|---|---|---|"]
    methods = ["grad_x_input", "integrated_gradients", "occlusion", "shapley_sampling", "lime_style"]
    for m in methods:
        cells = [
            f"{sum(r['timing_seconds'][f'{a}/{m}']) / len(r['timing_seconds'][f'{a}/{m}']):.1f} s"
            for a in r["config"]["archs"]
        ]
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_data():
    d = load("dataset.json")
    return (
        f"| rows used | channels | target | train / val / test rows | test windows | lookback → horizon |\n|---|---|---|---|---|---|\n"
        f"| {d['rows_used']:,} | {len(d['channels'])} ({', '.join(d['channels'])}) | {d['target']} | "
        f"{d['split_rows']['train']:,} / {d['split_rows']['val']:,} / {d['split_rows']['test']:,} | {d['test_windows']:,} | "
        f"{d['lookback']} h → {d['horizon']} h |"
    )


TABLES = {
    "data": table_data,
    "accuracy": table_accuracy,
    "main": table_main,
    "sanity": table_sanity,
    "agreement": table_agreement,
    "early": table_early,
    "timing": table_timing,
    "pinned": table_pinned,
}


def main() -> None:
    for path in TARGETS:
        if not path.exists():
            continue
        text = path.read_text()
        for name, fn in TABLES.items():
            pat = re.compile(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)", re.S)
            if pat.search(text):
                text = pat.sub(lambda m, fn=fn: m.group(1) + fn() + m.group(2), text)
        path.write_text(text)


if __name__ == "__main__":
    main()
