from ts_attribution_faithfulness.summary import aggregate_faithfulness, mean_std, paired_difference, paired_vs_random


def _rec(arch, method, seed, gap, base="mean"):
    return {
        "arch": arch,
        "method": method,
        "seed": seed,
        "eval_baseline": base,
        "deletion_auc": gap,
        "insertion_auc": 0.0,
        "gap": gap,
        "gap_per_window": [gap, gap],
    }


def test_mean_std():
    m, s = mean_std([1.0, 3.0])
    assert m == 2.0 and abs(s - 2**0.5) < 1e-9


def test_aggregate():
    recs = [_rec("a", "m", seed, float(seed + 1)) for seed in (0, 1, 2)]
    assert aggregate_faithfulness(recs)[("a", "m", "mean")]["gap"] == (2.0, 1.0)


def test_paired_difference_interval():
    recs = [_rec("a", "m", 0, 2.0), _rec("a", "ref", 0, 1.0)]
    mean, lo, hi = paired_difference(recs, "a", "m", "ref", "mean", n_boot=50)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0


def test_paired_vs_random():
    recs = [_rec("a", "m", 0, 2.0), _rec("a", "random", 0, 1.0)]
    assert paired_vs_random(recs, "a", "m", "mean") == 1.0
