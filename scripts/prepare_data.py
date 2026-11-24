#!/usr/bin/env python3
"""Download ETTh1.csv (2.6 MB) and write dataset statistics to results/metrics/dataset.json."""

from ts_attribution_faithfulness.common import CSV, config, write_json
from ts_attribution_faithfulness.data import N_TEST, N_TRAIN, N_VAL, download, load_splits, make_windows


def main() -> None:
    cfg = config()
    download(CSV)
    s = load_splits(CSV, cfg["lookback"])
    xt, _ = make_windows(s.test, cfg["lookback"], cfg["horizon"], s.target_idx)
    write_json(
        "dataset.json",
        {
            "rows_used": N_TRAIN + N_VAL + N_TEST,
            "channels": s.columns,
            "target": s.columns[s.target_idx],
            "split_rows": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
            "test_windows": len(xt),
            "target_mean_train": float(s.mean[s.target_idx]),
            "target_std_train": float(s.std[s.target_idx]),
            "lookback": cfg["lookback"],
            "horizon": cfg["horizon"],
        },
    )
    print("ok", CSV)


if __name__ == "__main__":
    main()
