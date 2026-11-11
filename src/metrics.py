"""Classification metrics and the statistics used for every comparison in the paper."""
import numpy as np
from scipy import stats


def accuracy(y, p):
    return float((p.argmax(1) == y).mean())


def macro_f1(y, pred, n_cls=None):
    n_cls = n_cls or int(max(y.max(), pred.max()) + 1)
    f = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def summary(y, p):
    p = p.astype(np.float64)
    p = p / p.sum(1, keepdims=True)
    return {"acc": accuracy(y, p), "f1": macro_f1(y, p.argmax(1), p.shape[1]), "ece": ece(y, p)}


def bootstrap_ci(y, pred, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def mcnemar(y, pa, pb):
    """Exact (binomial) McNemar test on paired predictions."""
    a, b = pa == y, pb == y
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    pval = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only": n01, "b_only": n10, "p": pval}


def paired(a, b):
    """Paired comparison over matched (budget, seed) runs: t-test, Wilcoxon, Cohen's d_z."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = 1.0
    return {"mean_diff": float(d.mean()), "t_p": float(t.pvalue), "wilcoxon_p": float(w),
            "d_z": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf"),
            "wins": int((d > 0).sum()), "n": int(len(d))}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
