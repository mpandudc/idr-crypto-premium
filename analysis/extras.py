"""Additional checks added after the full estimators (Nov 2026).

  stress     RQ4 on gaps that end in rupiah stress (top 5% of JISDOR depreciation), as Section III.4.5 promises
  tax_clean  threshold model for the PMK 50/2025 regime before the April 2026 break (Aug 2025 to Mar 2026)
  eth        variance decomposition of the total premium computed with Ethereum instead of Bitcoin
  cost       rupiah value of the premium on domestic Bitcoin trading, by year
  replay     when the L2 rule and the benchmark would have alerted during 2021 (in-sample illustration)
"""
import json

import numpy as np
import pandas as pd

from analysis.common import hac_ols, record
from analysis.final import tar_full
from analysis.rq4 import fixing_gaps
from config import BP, RESULTS


def stress(H, J):
    G = fixing_gaps(H, J)
    G = G[G.a >= "2022-01-01"].copy()
    cut = G.dj.quantile(0.95)
    G["stress"] = G.dj > cut
    rows = []
    for lab, g in (("stress", G[G.stress]), ("other", G[~G.stress]), ("all_2022+", G)):
        m = hac_ols(g.dj, g.du)
        rows.append(dict(sample=lab, n=len(g), slope=m.params.du, t=m.tvalues.du, r2=m.rsquared,
                         sign_match=float((np.sign(g.du) == np.sign(g.dj)).mean()),
                         mean_dj_bp=g.dj.mean() * BP, mean_du_bp=g.du.mean() * BP))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "extra_rq4_stress.csv", index=False)
    print(R.round(3).to_string(index=False))
    record("extra.stress.cut_bp", round(cut * BP, 1))
    for _, r in R.iterrows():
        for c in ("n", "slope", "t", "r2", "sign_match", "mean_dj_bp", "mean_du_bp"):
            record(f"extra.stress.{r['sample']}.{c}", round(float(r[c]), 3))


def tax_clean(D):
    x = D.loc["2025-08-01":"2026-03-31", "pt"]
    r = tar_full(x)
    m = x.median()
    out = dict(days=len(x), gL=r["gL"] * BP, gU=r["gU"] * BP, p_boot=r["p_boot"], median=m * BP,
               asym=(r["gU"] + r["gL"] - 2 * m) * BP, gL_lo=r["gL_lo"] * BP, gL_hi=r["gL_hi"] * BP,
               gU_lo=r["gU_lo"] * BP, gU_hi=r["gU_hi"] * BP)
    pd.Series(out).to_csv(RESULTS / "extra_tax_clean.csv")
    print({k: round(float(v), 3) for k, v in out.items()})
    for k, v in out.items():
        record(f"extra.tax_clean.{k}", round(float(v), 3))


def eth(D):
    g0 = D.dropna(subset=["pte", "pce", "pr"])
    rows = []
    for y, g in list(g0.groupby(g0.index.year)) + [("all", g0)]:
        v = g.pte.var()
        rows.append(dict(year=y, n=len(g), mean_total_bp=g.pte.mean() * BP,
                         share_rupiah=np.cov(g.pr, g.pte)[0, 1] / v, share_crypto=np.cov(g.pce, g.pte)[0, 1] / v))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "extra_eth_decomposition.csv", index=False)
    print(R.round(3).to_string(index=False))
    for _, r in R.iterrows():
        record(f"extra.eth.{r.year}.share_rupiah", round(float(r.share_rupiah), 3))
        record(f"extra.eth.{r.year}.mean_total_bp", round(float(r.mean_total_bp), 1))


def cost(H, D):
    """Premium times rupiah value traded on BTC/IDR, hour by hour, using the total premium of the same
    fixing day. This is the gross amount by which rupiah prices exceeded the global equivalent on traded
    volume (net of discount days), not a welfare loss."""
    X = H[["btcidr", "btcv"]].dropna().copy()
    day = X.index.normalize()
    X["pt"] = D.pt.reindex(day).ffill().values
    X["idr"] = X.btcidr * X.btcv
    X = X.dropna()
    X["paid"] = X.pt * X.idr
    R = X.groupby(X.index.year).agg(volume_idr_bn=("idr", lambda s: s.sum() / 1e9),
                                    premium_paid_idr_bn=("paid", lambda s: s.sum() / 1e9))
    R["volume_weighted_premium_bp"] = R.premium_paid_idr_bn / R.volume_idr_bn * BP
    R.to_csv(RESULTS / "extra_premium_cost.csv")
    print(R.round(1).to_string())
    for y, r in R.iterrows():
        record(f"extra.cost.{y}.volume_idr_bn", round(float(r.volume_idr_bn), 0))
        record(f"extra.cost.{y}.premium_paid_idr_bn", round(float(r.premium_paid_idr_bn), 1))
        record(f"extra.cost.{y}.vw_premium_bp", round(float(r.volume_weighted_premium_bp), 1))


def replay(D):
    """2021 replay with the dry-run frozen thresholds: in-sample, illustration only."""
    fz = json.loads((RESULTS / "frozen_rules_dryrun.json").read_text())
    c = fz["rules"]["L2 level and rising"]
    X = D.loc["2021"].copy()
    X["dpt"] = D.pt.diff().reindex(X.index)
    X["l2"] = (X.pt.abs() > c) & (np.sign(X.pt) * X.dpt > 0)
    X["bench"] = X.pt.abs() > 0.005
    X["red"] = X.pt.abs() > fz["episode_threshold"]
    m = X.resample("ME").agg(days=("pt", "size"), red_days=("red", "sum"), l2_alerts=("l2", "sum"),
                             bench_alerts=("bench", "sum"))
    m["mean_bp"] = X.pt.resample("ME").mean() * BP
    m.index = m.index.strftime("%Y-%m")
    m.to_csv(RESULTS / "extra_replay_2021.csv")
    print(m.round(1).to_string())
    record("extra.replay.l2_threshold", c)
    record("extra.replay.l2_alerts_2021", int(X.l2.sum()))
    record("extra.replay.bench_alerts_2021", int(X.bench.sum()))
    record("extra.replay.red_days_2021", int(X.red.sum()))
    if X.red.any():
        record("extra.replay.first_red", str(X.index[X.red][0].date()))
    if X.l2.any():
        record("extra.replay.first_l2", str(X.index[X.l2][0].date()))


def run(H, D, J):
    for name, f in (("stress", lambda: stress(H, J)), ("tax_clean", lambda: tax_clean(D)), ("eth", lambda: eth(D)),
                    ("cost", lambda: cost(H, D)), ("replay", lambda: replay(D))):
        print(f"--- {name}", flush=True)
        f()


def sample_report(H, D, A, month="2026-10"):
    """Numbers for the sample monthly price competitiveness report (Section IV.3), from public data only."""
    from config import COINS
    fz = json.loads((RESULTS / "frozen_rules_dryrun.json").read_text())
    m0 = pd.Period(month, "M")
    prev, yago = str(m0 - 1), str(m0 - 12)
    out = {}
    for lab, mo in (("this", month), ("prev", prev), ("yago", yago)):
        s = D.loc[mo]
        X = H.loc[mo, ["btcidr", "btcv"]].dropna()
        pt = D.pt.reindex(X.index.normalize()).ffill().values
        out[lab] = dict(days=len(s), total_bp=s.pt.mean() * BP, crypto_bp=s.pc.mean() * BP, rupiah_bp=s.pr.mean() * BP,
                        rupiah_share=s.pt.cov(s.pr) / s.pt.var() if len(s) > 2 else np.nan,
                        max_total_bp=s.pt.max() * BP, min_total_bp=s.pt.min() * BP,
                        volume_idr_bn=float((X.btcidr * X.btcv).sum() / 1e9),
                        premium_idr_bn=float((pt * X.btcidr * X.btcv).sum() / 1e9))
    # fixing gaps that end in the month
    G = fixing_gaps(H, D.jis)
    G = G[(G.b >= m0.start_time) & (G.b <= m0.end_time)]
    gaps = G.groupby("type").agg(n=("dj", "size"), dj_bp=("dj", lambda x: x.mean() * BP),
                                 du_bp=("du", lambda x: x.mean() * BP),
                                 same_sign=("dj", lambda x: float((np.sign(x) == np.sign(G.loc[x.index, "du"])).mean())))
    # widest crypto channels across assets in the month (median absolute, traded hours)
    rows = []
    u = H.usdtidr
    for c in COINS:
        p, v, g = (H.btcidr, H.btcv, H.glob) if c == "BTC" else (A.get(f"{c}_p"), A.get(f"{c}_v"), A.get(f"{c}_g"))
        if p is None:
            continue
        X = pd.DataFrame({"p": p, "v": v, "u": u, "g": g}).loc[month].dropna()
        X = X[X.v > 0]
        if len(X) < 50:
            continue
        pi = (np.log(X.p) - np.log(X.u) - np.log(X.g)) * BP
        rows.append(dict(asset=c, hours=len(X), median_abs_bp=pi.abs().median(), volume_idr_bn=float((X.p * X.v).sum() / 1e9)))
    W = pd.DataFrame(rows).sort_values("median_abs_bp", ascending=False)
    # alerts with the dry-run thresholds
    s = D.loc[month].copy()
    dpt = D.pt.diff().reindex(s.index)
    c2 = fz["rules"]["L2 level and rising"]
    l2 = s.index[(s.pt.abs() > c2) & (np.sign(s.pt) * dpt > 0)]
    bench = s.index[s.pt.abs() > 0.005]
    red = s.index[s.pt.abs() > fz["episode_threshold"]]
    hc = H.loc[month, "pc"]
    rep = dict(month=month, levels=out, gaps=gaps.reset_index().to_dict("records"),
               widest=W.head(10).to_dict("records"), l2_alerts=[str(d.date()) for d in l2],
               bench_alerts=[str(d.date()) for d in bench], red_days=[str(d.date()) for d in red],
               crypto_hourly_max_bp=float(hc.max() * BP), crypto_hourly_max_at=str(hc.idxmax()))
    (RESULTS / f"sample_report_{month}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float)[:4000])
