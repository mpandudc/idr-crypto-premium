"""RQ3 full version: pooled VECM per month and session, with day-block bootstrap intervals.

The simplified run (rq3.py) fits a separate VECM to every eight-hour day-session, and about
half of those fits are inadmissible. Here the day-sessions of one month and one session are
pooled into a single estimate:

  * the cointegrating vector is fixed at (1, -1): both series are the log price of Bitcoin in
    USDT, so the long-run relation is the domestic premium itself;
  * each day-session keeps its own mean (fixed effect), so a premium that sits at a different
    level on different days does not bias the adjustment coefficients;
  * lags never cross a session boundary or a missing minute;
  * the US session (23:00 to 07:00 WIB) is dated by the WIB day on which it starts;
  * the same pooling is repeated by year and session, by year, by session and for weekends.

Intervals come from resampling whole day-sessions with replacement (999 draws). Draws in which
the adjustment coefficients have the same sign (no admissible share) are counted and left out
of the percentiles.
"""
import numpy as np
import pandas as pd

from analysis.common import record
from config import INTERIM, M1_MONTHS, RESULTS

K = 5          # lagged differences, as in rq3.py
REPS = 999
MIN_OBS = 300  # usable minutes per day-session, as in rq3.py


def month_frame(mo):
    f = INTERIM / "m1" / f"{mo}.csv"
    if not f.exists():
        return None
    a = pd.Timestamp(mo + "-01")
    b = a + pd.offsets.MonthBegin(1)
    m = pd.read_csv(f, index_col=0, parse_dates=True)
    idx = pd.date_range(a, b, freq="1min", inclusive="left")
    X = pd.DataFrame({"bi": m.bi.reindex(idx).ffill(limit=5), "ui": m.ui.reindex(idx).ffill(limit=30),
                      "g": m.g.reindex(idx)}, index=idx)
    X["dom"] = np.log(X.bi / X.ui)
    X["glob"] = np.log(X.g)
    wib = X.index + pd.Timedelta(hours=7)
    X["sess"] = np.select([(wib.hour >= 7) & (wib.hour < 15), (wib.hour >= 15) & (wib.hour < 23)],
                          ["Asia", "Europe"], "US")
    X["day"] = (wib - pd.Timedelta(hours=7)).normalize()
    X["weekend"] = X.day.dt.dayofweek >= 5
    return X[["dom", "glob", "sess", "day", "weekend"]]


def block_stats(x):
    """Sufficient statistics of one day-session: X'X, X'Y, Y'Y and n, with the fixed effect
    removed by demeaning within the block. x: (n, 2) log prices on a regular one-minute grid,
    NaN where missing."""
    dp = np.diff(x, axis=0)                      # dp[t] = p[t+1] - p[t]
    ect = (x[:, 0] - x[:, 1])[:-1]               # premium at t
    cols = [ect] + [np.r_[np.full(i, np.nan), dp[:-i, j]] for i in range(1, K + 1) for j in (0, 1)]
    R = np.column_stack(cols)
    ok = ~np.isnan(R).any(1) & ~np.isnan(dp).any(1)
    if ok.sum() < MIN_OBS:
        return None
    R, Y = R[ok], dp[ok]
    R = R - R.mean(0)
    Y = Y - Y.mean(0)
    return R.T @ R, R.T @ Y, Y.T @ Y, len(Y)


def shares(a, om):
    """Price-discovery shares of the domestic market from alpha (2,) and Omega (2, 2), with the
    same definitions as common.vecm_shares; None if inadmissible."""
    ap = np.array([a[1], -a[0]])
    if ap.sum() == 0:
        return None
    cs = ap / ap.sum()
    if not (0 < cs[0] < 1):
        return None
    out = []
    for order in ([0, 1], [1, 0]):
        P = np.eye(2)[order]
        F = np.linalg.cholesky(P @ om @ P.T)
        v = (ap[order] @ F) ** 2
        v = v / v.sum()
        o = np.empty(2)
        o[order] = v
        out.append(o[0])
    lo, hi = min(out), max(out)
    mid = (lo + hi) / 2
    if not (0 < mid < 1):
        return None
    il1 = abs(mid / (1 - mid) * cs[1] / cs[0])
    return dict(is_lo=lo, is_hi=hi, is_mid=mid, cs=cs[0], ils=il1 ** 2 / (il1 ** 2 + 1))


def estimate(XtX, XtY, YtY, n):
    B = np.linalg.solve(XtX, XtY)
    E = YtY - B.T @ XtY
    om = E / (n - XtX.shape[0])
    a = B[0]
    return a, om, shares(a, om)


def pooled(S, rng):
    """S: list of block statistics. Point estimate plus day-block bootstrap."""
    XtX = np.stack([s[0] for s in S]); XtY = np.stack([s[1] for s in S])
    YtY = np.stack([s[2] for s in S]); N = np.array([s[3] for s in S], dtype=float)
    a, om, r = estimate(XtX.sum(0), XtY.sum(0), YtY.sum(0), N.sum())
    row = dict(blocks=len(S), n=int(N.sum()), alpha_dom=a[0], alpha_glob=a[1])
    row.update(r or dict(is_lo=np.nan, is_hi=np.nan, is_mid=np.nan, cs=np.nan, ils=np.nan))
    ils, cs = [], []
    for _ in range(REPS):
        w = np.bincount(rng.integers(0, len(S), len(S)), minlength=len(S)).astype(float)
        rb = estimate(np.tensordot(w, XtX, 1), np.tensordot(w, XtY, 1), np.tensordot(w, YtY, 1), w @ N)[2]
        if rb:
            ils.append(rb["ils"]); cs.append(rb["cs"])
    row["admissible_draws"] = len(ils) / REPS
    if len(ils) >= 50:
        row["ils_lo"], row["ils_hi"] = np.percentile(ils, [5, 95])
        row["cs_lo"], row["cs_hi"] = np.percentile(cs, [5, 95])
        row["p_ils_above_half"] = float(np.mean(np.array(ils) >= 0.5))
    return row


def run():
    rng = np.random.default_rng(11)
    blocks = []                                   # (month, year, sess, weekend, stats)
    for mo in M1_MONTHS:
        X = month_frame(mo)
        if X is None:
            continue
        for (day, s), x in X.groupby(["day", "sess"]):
            st = block_stats(x[["dom", "glob"]].values)
            if st:
                blocks.append((mo, int(mo[:4]), s, bool(x.weekend.iloc[0]), st))
        print("  rq3 pooled", mo, len(blocks), flush=True)
    B = pd.DataFrame([b[:4] for b in blocks], columns=["month", "year", "sess", "weekend"])
    st = [b[4] for b in blocks]

    groups = []
    for (mo, s), g in B.groupby(["month", "sess"]):
        groups.append(("month_sess", f"{mo} {s}", g.index))
    for (y, s), g in B.groupby(["year", "sess"]):
        groups.append(("year_sess", f"{y} {s}", g.index))
    for y, g in B.groupby("year"):
        groups.append(("year", str(y), g.index))
    for s, g in B.groupby("sess"):
        groups.append(("sess", s, g.index))
    groups.append(("weekend", "weekend", B.index[B.weekend]))
    groups.append(("weekend", "weekday", B.index[~B.weekend]))
    groups.append(("all", "all", B.index))

    rows = []
    for level, key, ix in groups:
        r = pooled([st[i] for i in ix], rng)
        r.update(level=level, key=key)
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "rq3_pooled.csv", index=False)
    show = ["level", "key", "blocks", "ils", "ils_lo", "ils_hi", "cs", "is_lo", "is_hi", "admissible_draws"]
    print(R[R.level != "month_sess"][show].round(3).to_string(index=False), flush=True)

    M = R[R.level == "month_sess"]
    record("final.rq3.day_sessions", len(blocks))
    record("final.rq3.month_sessions", len(M))
    record("final.rq3.month_sessions_admissible", int(M.ils.notna().sum()))
    record("final.rq3.month_sessions_ils_below_half", round(float((M.ils.dropna() < 0.5).mean()), 3))
    record("final.rq3.month_sessions_ci_below_half", int((M.ils_hi < 0.5).sum()))
    record("final.rq3.month_sessions_ci_above_half", int((M.ils_lo > 0.5).sum()))
    for _, r in R[R.level != "month_sess"].iterrows():
        k = r.key.replace(" ", ".")
        for c in ("ils", "ils_lo", "ils_hi", "cs"):
            if pd.notna(r.get(c)):
                record(f"final.rq3.{r.level}.{k}.{c}", round(float(r[c]), 3))
