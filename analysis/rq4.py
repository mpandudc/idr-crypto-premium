"""RQ4 USDT/IDR ahead of the JISDOR fixing, plus the reconciliation with earlier evidence."""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from analysis.common import hac_ols, record, vecm_shares
from config import BP, RESULTS, START


def fixing_gaps(H, J):
    """Change in JISDOR between consecutive fixings against the change in USDT/IDR from
    11:00 WIB on the first fixing day to 09:00 WIB on the next one (hourly bars are
    labelled by opening time: b + 1h is the bar that closes at 09:00 WIB)."""
    ui, fx = H.usdtidr, H.fx if "fx" in H else pd.Series(dtype=float)
    rows = []
    jd = J[J.index >= START].index
    for a, b in zip(jd[:-1], jd[1:]):
        u0, u1 = ui.asof(a + pd.Timedelta(hours=3)), ui.asof(b + pd.Timedelta(hours=1))
        f0, f1 = (fx.asof(a + pd.Timedelta(hours=3)), fx.asof(b + pd.Timedelta(hours=1))) if len(fx.dropna()) else (np.nan, np.nan)
        rows.append(dict(a=a, b=b, gap=(b - a).days, dj=np.log(J[b] / J[a]), du=np.log(u1 / u0),
                         df=np.log(f1 / f0) if np.isfinite([f0, f1]).all() else np.nan))
    G = pd.DataFrame(rows).dropna(subset=["du"])
    G["type"] = np.select([G.gap == 1, G.gap.between(2, 3)], ["overnight", "weekend"], "holiday")
    return G


def _fit(g, label):
    m = hac_ols(g.dj, g.du)
    return dict(sample=label, n=len(g), slope=m.params.du, t=m.tvalues.du, r2=m.rsquared,
                sign_match=(np.sign(g.du) == np.sign(g.dj)).mean())


def run(H, D, J):
    G = fixing_gaps(H, J)
    G.to_csv(RESULTS / "rq4_gaps.csv", index=False)
    rows = [_fit(g, t) for t, g in G.groupby("type")] + [_fit(G, "all")]
    rows += [_fit(g, str(y)) for y, g in G.groupby(G.a.dt.year)]
    late = G[G.a >= "2024-01-01"]
    rows += [_fit(g, f"2024+ {t}") for t, g in late.groupby("type")]
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "rq4_fixing.csv", index=False)
    print(T.round(3).to_string(index=False))
    for _, r in T.iterrows():
        k = r["sample"].replace(" ", "_")
        record(f"rq4.{k}.slope", round(r.slope, 3))
        record(f"rq4.{k}.r2", round(r.r2, 3))
        record(f"rq4.{k}.n", int(r.n))

    # competing predictor: market USD/IDR quote over the same interval (from 2024)
    g = G.dropna(subset=["df"])
    g = g[g.a >= "2024-01-01"]
    if len(g) > 30:
        m1 = hac_ols(g.dj, g[["df"]])
        m2 = hac_ols(g.dj, g[["df", "du"]])
        record("rq4.competing.n", len(g))
        record("rq4.competing.r2_quote", round(m1.rsquared, 3))
        record("rq4.competing.r2_quote_plus_usdt", round(m2.rsquared, 3))
        record("rq4.competing.t_usdt", round(m2.tvalues["du"], 2))
        print("competing: R2 quote", round(m1.rsquared, 3), "quote+USDT", round(m2.rsquared, 3), "n", len(g))
    hol = G[G.type == "holiday"][["a", "b", "gap", "du", "dj"]].copy()
    hol[["du", "dj"]] *= BP
    hol.to_csv(RESULTS / "rq4_holidays.csv", index=False)

    # daily VECM: USDT/IDR (in USD terms) against JISDOR
    rows = []
    for y, s in D.groupby(D.index.year):
        r = vecm_shares(np.column_stack([np.log(s.usdtidr / s.u), np.log(s.jis)]), k=2)
        if r:
            r["year"] = y
            rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "rq4_daily_vecm.csv", index=False)

    # reconciliation with Maulana and Gitayuda (2026): 2024 onward
    X = D.loc["2024":]
    L = pd.DataFrame({"idr": np.log(X.btcidr), "usd": np.log(X.glob), "fx": np.log(X.jis * X.u)})
    L["conv"] = L.usd + L.fx
    rec = {"days": len(X)}
    for cols, name in ((["idr", "usd"], "no_fx"), (["idr", "conv"], "converted")):
        r = coint_johansen(L[cols], 0, 1)
        rec[f"trace_{name}"] = r.lr1[0]
        rec["cv5"] = r.cvt[0, 1]
    rec["adf_p_jisdor"] = adfuller(L.fx)[1]
    R = L[["idr", "usd"]].diff().dropna() * 100
    for o in (["idr", "usd"], ["usd", "idr"]):
        f = VAR(R[o]).fit(maxlags=5, ic="aic").fevd(10).decomp
        i = o.index("idr")
        rec[f"fevd_own_{o[0]}_first"] = f[i, -1, i] * 100
    rec["corr_returns"] = R.corr().iloc[0, 1]
    pd.Series(rec).to_csv(RESULTS / "reconciliation.csv")
    for k, v in rec.items():
        record(f"recon.{k}", round(float(v), 3))
    print("reconciliation", {k: round(float(v), 2) for k, v in rec.items()})
