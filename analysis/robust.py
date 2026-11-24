"""Robustness checks promised in Section III.4 of the thesis.

  global_cb   total premium and decomposition with Coinbase BTC-USD as the global price
  calm        size, persistence, decomposition and the band model without the 1% most volatile days
  cutoff08    fixing-to-fixing regressions with USDT/IDR read at 08:00 WIB instead of 09:00
  gfevd       generalised forecast error variance decomposition (Pesaran and Shin, 1998) for the
              reconciliation with Maulana and Gitayuda (2026), Bitcoin and Ethereum, 2024 onward
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

import sources as src
from analysis.common import half_life, hac_ols, record
from analysis.rq4 import fixing_gaps
from analysis.rq5 import tar_test
from config import BP, END, RESULTS, START


def _shares(s, pt="pt", pr="pr"):
    return s[pt].cov(s[pr]) / s[pt].var()


def global_cb(H, D):
    cb = src.coinbase("BTC-USD", START, END).c
    if cb.index.tz is not None:
        cb.index = cb.index.tz_convert(None)
    X = D.copy()
    X["cb"] = cb.reindex(X.index + pd.Timedelta(hours=2)).values   # same 02:00 UTC bar as the daily panel
    X = X.dropna(subset=["cb"])
    X["pt_cb"] = np.log(X.btcidr) - np.log(X.cb * X.jis)
    rows = []
    for y, s in list(X.groupby(X.index.year)) + [("all", X)]:
        rows.append(dict(year=y, n=len(s), mean_bp=s.pt.mean() * BP, mean_cb_bp=s.pt_cb.mean() * BP,
                         corr=s.pt.corr(s.pt_cb), share_rupiah=_shares(s), share_rupiah_cb=_shares(s, "pt_cb"),
                         half_life_cb=half_life(s.pt_cb)))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "robust_global_cb.csv", index=False)
    print(R.round(3).to_string(index=False))
    for _, r in R.iterrows():
        for c in ("mean_cb_bp", "corr", "share_rupiah_cb"):
            record(f"robust.cb.{r.year}.{c}", round(float(r[c]), 3))


def calm(D):
    X = D[D.rv < D.rv.quantile(0.99)]
    rows = []
    for y, s in list(X.groupby(X.index.year)) + [("all", X)]:
        rows.append(dict(year=y, n=len(s), mean_bp=s.pt.mean() * BP, share_rupiah=_shares(s),
                         half_life=half_life(s.pt)))
    R = pd.DataFrame(rows)
    t = tar_test(X.pt)
    R.to_csv(RESULTS / "robust_calm.csv", index=False)
    print(R.round(3).to_string(index=False))
    print("TAR without top 1% days", round(t["gL"] * BP), round(t["gU"] * BP), "p", t["p_boot"])
    record("robust.calm.days_dropped", int(len(D) - len(X)))
    record("robust.calm.tar_gL", round(t["gL"] * BP, 1))
    record("robust.calm.tar_gU", round(t["gU"] * BP, 1))
    record("robust.calm.tar_p", round(t["p_boot"], 3))
    for _, r in R.iterrows():
        record(f"robust.calm.{r.year}.mean_bp", round(float(r.mean_bp), 1))
        record(f"robust.calm.{r.year}.share_rupiah", round(float(r.share_rupiah), 3))
        record(f"robust.calm.{r.year}.half_life", round(float(r.half_life), 2))


def cutoff08(H, J):
    G = fixing_gaps(H, J, end_hours=0)              # bar opening 00:00 UTC closes at 08:00 WIB
    G = G[G.a >= "2024-01-01"]
    rows = []
    for t, g in G.groupby("type"):
        m = hac_ols(g.dj, g.du)
        rows.append(dict(type=t, n=len(g), slope=m.params.du, t=m.tvalues.du, r2=m.rsquared))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "robust_cutoff08.csv", index=False)
    print(R.round(3).to_string(index=False))
    for _, r in R.iterrows():
        for c in ("slope", "t", "r2"):
            record(f"robust.cutoff08.{r.type}.{c}", round(float(r[c]), 3))


def _gfevd(res, h=10):
    """Generalised FEVD at horizon h: share of variable i's forecast error variance due to j,
    rows normalised to one."""
    S = np.asarray(res.sigma_u)
    Phi = res.ma_rep(h - 1)
    k = S.shape[0]
    th = np.zeros((k, k))
    for i in range(k):
        den = sum(Phi[l][i] @ S @ Phi[l][i] for l in range(h))
        for j in range(k):
            th[i, j] = sum((Phi[l][i] @ S[:, j]) ** 2 for l in range(h)) / S[j, j] / den
    return th / th.sum(1, keepdims=True)


def gfevd(D):
    X = D.loc["2024":]
    rows = []
    for asset, dom, glob in (("BTC", "btcidr", "glob"), ("ETH", "ethidr", "eglob")):
        R = pd.DataFrame({"idr": np.log(X[dom]), "usd": np.log(X[glob])}).diff().dropna() * 100
        res = VAR(R.values).fit(maxlags=5, ic="aic")
        th = _gfevd(res)
        rows.append(dict(asset=asset, lags=res.k_ar, own_share_idr=th[0, 0] * 100, global_share_idr=th[0, 1] * 100,
                         corr=R.corr().iloc[0, 1]))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "robust_gfevd.csv", index=False)
    print(T.round(2).to_string(index=False))
    for _, r in T.iterrows():
        record(f"robust.gfevd.{r.asset}.own_share_idr", round(float(r.own_share_idr), 1))
        record(f"robust.gfevd.{r.asset}.global_share_idr", round(float(r.global_share_idr), 1))


def run(H, D, J):
    for name, f in (("calm", lambda: calm(D)), ("cutoff08", lambda: cutoff08(H, J)), ("gfevd", lambda: gfevd(D)),
                    ("global_cb", lambda: global_cb(H, D))):
        print(f"--- {name}", flush=True)
        f()
