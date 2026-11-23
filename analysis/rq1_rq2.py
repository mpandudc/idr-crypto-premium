"""RQ1 size and persistence of the premium; RQ2 decomposition into crypto and rupiah channels."""
import numpy as np
import pandas as pd

from analysis.common import half_life, record
from config import BP, RESULTS


def run(H, D, J):
    rows = []
    for y, s in D.groupby(D.index.year):
        hy = H.loc[str(y), "pc"]
        v = s.pt.var()
        rows.append(dict(year=y, days=len(s), total_mean=s.pt.mean() * BP, total_med=s.pt.median() * BP,
                         p5=s.pt.quantile(.05) * BP, p95=s.pt.quantile(.95) * BP,
                         crypto_mean=s.pc.mean() * BP, rupiah_mean=s.pr.mean() * BP,
                         rupiah_share=s.pt.cov(s.pr) / v, share_pos=(s.pt > 0).mean(),
                         hl_total_days=half_life(s.pt), hl_rupiah_days=half_life(s.pr),
                         hl_crypto_hours=half_life(hy), crypto_in25_hours=(hy.abs() < .0025).mean()))
    T = pd.DataFrame(rows).set_index("year")
    T.to_csv(RESULTS / "rq1_by_year.csv")
    print(T.round(2).to_string())
    for y, r in T.iterrows():
        record(f"rq1.{y}.days", int(r.days))
        record(f"rq1.{y}.total_mean", round(r.total_mean, 1))
        record(f"rq1.{y}.p5", round(r.p5, 1))
        record(f"rq1.{y}.p95", round(r.p95, 1))
        record(f"rq1.{y}.crypto_mean", round(r.crypto_mean, 1))
        record(f"rq1.{y}.rupiah_mean", round(r.rupiah_mean, 1))
        record(f"rq1.{y}.rupiah_share", round(r.rupiah_share, 3))
        record(f"rq1.{y}.hl_crypto_hours", round(r.hl_crypto_hours, 2))
    record("rq1.days_total", int(len(D)))

    # market phase: 90-trading-day return of global BTC
    D = D.copy()
    D["phase"] = np.where(D.ret90 > 0, "bull", "bear")
    ph = D.groupby("phase").agg(days=("pt", "size"), total=("pt", lambda x: x.mean() * BP),
                                rupiah=("pr", lambda x: x.mean() * BP), crypto=("pc", lambda x: x.mean() * BP))
    ph["hl_total"] = [half_life(D.loc[D.phase == p, "pt"]) for p in ph.index]
    ph.to_csv(RESULTS / "rq1_phase.csv")

    # event windows (30 days either side)
    ev = {"Bull peak Apr 2021": "2021-04-14", "Bull peak Nov 2021": "2021-11-10",
          "Crypto tax 1 May 2022": "2022-05-01", "FTX 11 Nov 2022": "2022-11-11",
          "CFX launch 17 Jul 2023": "2023-07-17", "Halving 20 Apr 2024": "2024-04-20"}
    rows = []
    for k, v in ev.items():
        t = pd.Timestamp(v)
        a = D.loc[t - pd.Timedelta(days=30):t - pd.Timedelta(days=1)]
        b = D.loc[t:t + pd.Timedelta(days=30)]
        rows.append(dict(event=k, before_total=a.pt.mean() * BP, after_total=b.pt.mean() * BP,
                         before_rupiah=a.pr.mean() * BP, after_rupiah=b.pr.mean() * BP))
    E = pd.DataFrame(rows)
    E.to_csv(RESULTS / "rq1_events.csv", index=False)
    record("rq1.after_apr2021_peak_total", round(E.loc[0, "after_total"], 1))

    # scenario analogues (rupiah channel mean / 95th percentile)
    sc = {"open_rally_2024-02_03": ("2024-02-01", "2024-03-31"),
          "tight_rally_2021-02_04": ("2021-02-01", "2021-04-30"),
          "quiet_open_2023-04_09": ("2023-04-01", "2023-09-30"),
          "quiet_tight_2026-04_11": ("2026-04-01", "2026-11-30")}
    rows = []
    for k, (a, b) in sc.items():
        s = D.loc[a:b, "pr"] * BP
        rows.append(dict(scenario=k, days=len(s), rupiah_mean=s.mean(), rupiah_p95=s.quantile(.95)))
        record(f"scen.{k}.mean", round(s.mean(), 1))
        record(f"scen.{k}.p95", round(s.quantile(.95), 1))
    pd.DataFrame(rows).to_csv(RESULTS / "scenario_analogues.csv", index=False)

    # RQ2: covariance decomposition, var(pt) = cov(pt,pc) + cov(pt,pr)
    def shares(s):
        v = s.pt.var()
        return pd.Series(dict(crypto=s.pt.cov(s.pc) / v, rupiah=s.pt.cov(s.pr) / v, n=len(s)))

    T2 = D.groupby(D.index.year).apply(shares)
    T2.loc["all"] = shares(D)
    T2.loc["top10 BTC volatility"] = shares(D[D.rv >= D.rv.quantile(.9)])
    dj = np.log(D.jis).diff()
    T2.loc["top10 rupiah depreciation"] = shares(D[dj >= dj.quantile(.9)])
    e = D.dropna(subset=["pte"])
    T2.loc["ETH all"] = pd.Series(dict(crypto=e.pte.cov(e.pce) / e.pte.var(), rupiah=e.pte.cov(e.pr) / e.pte.var(), n=len(e)))
    T2.to_csv(RESULTS / "rq2_decomposition.csv")
    print(T2.round(3).to_string())
    record("rq2.all.rupiah_share", round(T2.loc["all", "rupiah"], 3))
    record("rq2.eth.rupiah_share", round(T2.loc["ETH all", "rupiah"], 3))
