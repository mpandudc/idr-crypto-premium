"""Estimators shared across research questions."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM

from config import RESULTS

SUMMARY = {}


def record(key, value):
    """Store a headline number for results/summary.json (used by verify.py)."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    SUMMARY[key] = value


def save_summary():
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "summary.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(SUMMARY)
    p.write_text(json.dumps(old, indent=1, default=float))


def half_life(x, min_n=50):
    """AR(1) half-life in units of the series' frequency."""
    x = pd.Series(x).dropna()
    if len(x) < min_n:
        return np.nan
    r = np.polyfit(x.values[:-1], x.values[1:], 1)[0]
    return np.log(0.5) / np.log(r) if 0 < r < 1 else np.nan


def hac_ols(y, X, lags=3):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def vecm_shares(X, k=3):
    """Two-variable VECM price-discovery shares for the first variable.
    Returns Hasbrouck information share bounds (is_lo, is_hi), Gonzalo-Granger component
    share (cs) and information leadership share (ils), or None if inadmissible."""
    try:
        m = VECM(X, k_ar_diff=k, coint_rank=1, deterministic="ci").fit()
    except Exception:
        return None
    a = m.alpha[:, 0]
    om = m.sigma_u
    ap = np.array([a[1], -a[0]])  # orthogonal complement of alpha
    if ap.sum() == 0:
        return None
    cs = ap / ap.sum()
    if not (0 < cs[0] < 1):
        return None

    def info_share(order):
        P = np.eye(2)[order]
        F = np.linalg.cholesky(P @ om @ P.T)
        v = (ap[order] @ F) ** 2
        v = v / v.sum()
        o = np.empty(2)
        o[order] = v
        return o

    i1, i2 = info_share([0, 1]), info_share([1, 0])
    lo, hi = min(i1[0], i2[0]), max(i1[0], i2[0])
    mid = (lo + hi) / 2
    if not (0 < mid < 1):
        return None
    il1 = abs(mid / (1 - mid) * cs[1] / cs[0])
    il2 = 1 / il1
    return dict(is_lo=lo, is_hi=hi, is_mid=mid, cs=cs[0], ils=il1 / (il1 + il2), n=len(X))


def domestic_tax(date):
    """Transaction tax on a domestic crypto (sale, purchase), in log points:
    none before PMK 68/2022; 0.1% income tax on sales and 0.11% VAT on purchases from 1 May 2022;
    0.21% on sales and no VAT from 1 Aug 2025 (PMK 50/2025)."""
    if date < pd.Timestamp("2022-05-01"):
        return 0.0, 0.0
    if date < pd.Timestamp("2025-08-01"):
        return 0.001, 0.0011
    return 0.0021, 0.0


def cost_band(date, professional=True, kind="crypto"):
    """Cost of closing a premium (upper, lower), in log points, crypto side only.
    Domestic fee 0 (professional) or 0.3% (retail) per domestic trade, global fee 0.1%, transfer 0.02%.

    kind="crypto": the crypto channel is closed with two domestic trades, one sale and one purchase
      (sell BTC and buy USDT when BTC/IDR is rich; buy BTC and sell USDT when it is cheap), so both
      directions pay both taxes and the band is symmetric.
    kind="total": the total premium is closed with one domestic trade and a rupiah conversion outside
      the platform. A positive premium is closed by selling domestically (sale tax), a negative one by
      buying domestically (purchase VAT), so the band is direction-specific. The rupiah conversion cost
      is not observed and is not included."""
    fd = 0.0 if professional else 0.003
    sale, buy = domestic_tax(date)
    base = 0.001 + 0.0002
    if kind == "crypto":
        c = base + 2 * fd + sale + buy
        return c, c
    return base + fd + sale, base + fd + buy


def episodes(pt, th):
    """Start dates (flag=1) and membership of runs of 2+ days with |pt| > th."""
    red = (pt.abs() > th).astype(int)
    grp = (red.diff() != 0).cumsum()
    runs = red.groupby(grp).agg(["first", "size"])
    good = runs.index[(runs["first"] == 1) & (runs["size"] >= 2)]
    start = pd.Series(0, index=pt.index)
    member = pd.Series(False, index=pt.index)
    for gi in good:
        ix = pt.index[(grp == gi).values]
        start[ix[0]] = 1
        member[ix] = True
    return start, member
