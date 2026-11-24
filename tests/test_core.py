import numpy as np
import torch

from ts_attribution_faithfulness import attribution as A
from ts_attribution_faithfulness import faithfulness as F
from ts_attribution_faithfulness.data import make_windows
from ts_attribution_faithfulness.models import ARCHS, LinearForecaster

L, C, H, BLOCK = 16, 3, 4, 4


def _lin(seed=0):
    torch.manual_seed(seed)
    m = LinearForecaster(L, C, H)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _setup(n=5):
    torch.manual_seed(1)
    x = torch.randn(n, L, C)
    gid, g = A.group_index(L, C, BLOCK)
    return x, torch.zeros_like(x), gid, g


def test_windows_shapes_and_alignment():
    s = np.arange(40, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
    x, y = make_windows(s, 6, 3, 1)
    assert x.shape == (32, 6, 2) and y.shape == (32, 3)
    assert x[0, -1, 0] == 5 and y[0, 0] == 6 and y[0, -1] == 8


def test_group_index_and_aggregate_preserve_sum():
    x, _, gid, g = _setup()
    assert g == (L // BLOCK) * C and int(gid.max()) == g - 1
    agg = A.aggregate(x, gid, g)
    assert torch.allclose(agg.sum(-1), x.flatten(1).sum(-1), atol=1e-5)


def test_models_output_shape():
    x = torch.randn(2, L, C)
    for name, cls in ARCHS.items():
        assert cls(L, C, H)(x).shape == (2, H), name


def test_tcn_blocks_are_causal():
    from ts_attribution_faithfulness.models import _Block

    blk = _Block(C, 8, dilation=4).eval()
    x = torch.randn(1, C, L)
    x2 = x.clone()
    x2[:, :, 10:] += 5.0  # change only the future
    assert torch.allclose(blk(x)[:, :, :10], blk(x2)[:, :, :10], atol=1e-6)


def test_linear_methods_match_exact():
    m = _lin()
    x, b, gid, g = _setup()
    exact = A.linear_exact(m, x, b, gid, g)
    for name in ("grad_x_input", "integrated_gradients", "occlusion", "shapley_sampling"):
        a = A.explain(name, m, x, b, gid, g)
        assert torch.allclose(a, exact, atol=1e-4), name


def test_lime_recovers_linear_attribution():
    m = _lin()
    x, b, gid, g = _setup()
    a = A.lime_style(m, x, b, gid, g, samples=2000, ridge=1e-6)
    assert torch.allclose(a, A.linear_exact(m, x, b, gid, g), atol=5e-2)


def test_integrated_gradients_completeness_nonlinear():
    torch.manual_seed(0)
    m = ARCHS["lstm"](L, C, H)
    for p in m.parameters():
        p.requires_grad_(False)
    x, b, gid, g = _setup(3)
    a = A.integrated_gradients(m, x, b, gid, g, steps=200)
    gap = A.scalar(m, x) - A.scalar(m, b)
    assert torch.allclose(a.sum(-1), gap, atol=2e-3)


def test_shapley_efficiency_nonlinear():
    torch.manual_seed(0)
    m = ARCHS["lstm"](L, C, H)
    x, b, gid, g = _setup(2)
    a = A.shapley_sampling(m, x, b, gid, g, perms=8)
    gap = A.scalar(m, x) - A.scalar(m, b)
    assert torch.allclose(a.sum(-1), gap, atol=1e-4)  # every permutation telescopes


def test_exact_ranking_beats_random_and_reversed():
    m = _lin()
    x, b, gid, g = _setup(8)
    exact = A.linear_exact(m, x, b, gid, g)
    sc = lambda a: F.scores(F.curves(m, x, b, a, gid))["gap"].mean()  # noqa: E731
    rnd = A.random_attribution(8, g)
    rev = 1.0 / (exact.abs() + 1e-3)  # least important first
    assert sc(exact) > sc(rnd) > sc(rev)


def test_curves_endpoints():
    m = _lin()
    x, b, gid, g = _setup(4)
    c = F.curves(m, x, b, A.linear_exact(m, x, b, gid, g), gid)
    assert torch.allclose(c["del"][:, 0], torch.zeros(4), atol=1e-6)  # nothing deleted
    assert torch.allclose(c["ins"][:, -1], torch.zeros(4), atol=1e-5)  # everything inserted


def test_spearman_and_jaccard():
    a = torch.tensor([[1.0, 2, 3, 4]])
    assert float(F.spearman(a, a)) == 1.0
    assert float(F.spearman(a, -a)) == -1.0
    assert float(F.topk_jaccard(a, a, 2)) == 1.0
    assert float(F.topk_jaccard(a, torch.tensor([[4.0, 3, 2, 1]]), 2)) == 0.0
    assert abs(F.rank_correlation([1, 2, 3], [1, 2, 3]) - 1) < 1e-6


def test_pinned_group_is_never_removed():
    m = _lin()
    x, b, gid, g = _setup(3)
    exact = A.linear_exact(m, x, b, gid, g)
    pin = int(exact.abs().mean(0).argmax())
    for mode in ("del", "ins"):
        masks = F.rank_masks(exact, mode, pin)
        assert masks.shape == (3, g, g)  # G-1 steps + 1
        assert torch.all(masks[:, :, pin] == 1)
    d = F.rank_masks(exact, "del", pin)
    assert torch.all(d[:, 0] == 1) and d[:, -1].sum() == 3  # only the pinned group remains at the end
