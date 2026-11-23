"""RQ6 decision rules: rolling-origin backtest by quarter with expanding calibration windows.
Each rule's threshold is chosen on the calibration window by expected loss
(cost ratio x missed episodes + false alarms) and then applied, frozen, to the next quarter."""
import numpy as np
import pandas as pd

from analysis.common import episodes, half_life, record
from config import EPISODE_TH, RESULTS

LEAD = 5  # a signal counts as a hit if it comes within 5 fixing days before an episode starts


def run(H, D, J):
    D = D.copy()
    u9 = H.usdtidr[H.index.hour == 1]  # 09:00 WIB, one hour before the fixing
    u9.index = u9.index.normalize()
    D["a7"] = np.log(u9.reindex(D.index)) - np.log(D.jis.shift(1))
    D["dpt"] = D.pt.diff()
    ep, inep = episodes(D.pt, EPISODE_TH)
    record("rq6.episodes_by_year", {int(k): int(v) for k, v in ep.groupby(ep.index.year).sum().items()})

    def evaluate(sig, idx):
        s = sig.reindex(idx).fillna(False).astype(bool).values
        e, ie = ep.reindex(idx).values, inep.reindex(idx).values
        starts = np.where(e == 1)[0]
        hits, leads = 0, []
        for k in starts:
            w = s[max(0, k - LEAD):k]
            if w.any():
                hits += 1
                leads.append(len(w) - np.where(w)[0][0])
        sp = np.where(s & ~ie)[0]
        fas = sum(0 if (e[j + 1:j + 1 + LEAD] == 1).any() else 1 for j in sp)
        return dict(eps=len(starts), hits=hits, signals=len(sp), fas=fas, leads=leads)

    def persistence(c, cal):
        h = max(1, int(round(half_life(cal.pt))))
        return (D.pt.abs() > c).rolling(h + 1).sum() >= h + 1

    makers = {
        "L1 level": (lambda c, cal: D.pt.abs() > c, [0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065]),
        "L2 level and rising": (lambda c, cal: (D.pt.abs() > c) & (np.sign(D.pt) * D.dpt > 0), [0.003, 0.004, 0.005, 0.006]),
        "L3 rupiah channel level": (lambda c, cal: D.pr.abs() > c, [0.003, 0.004, 0.005, 0.006]),
        "L4 overnight USDT/IDR vs last fixing": (lambda c, cal: D.a7.abs() > c, [0.003, 0.004, 0.005, 0.006, 0.008]),
        "P1 persistence beyond threshold": (persistence, [0.003, 0.004, 0.005, 0.006]),
    }
    out = []
    for ratio in (1, 3, 10):
        res = {k: dict(eps=0, hits=0, signals=0, fas=0, leads=[]) for k in list(makers) + ["BENCH 0.5%"]}
        for q in pd.period_range("2022Q1", "2026Q4", freq="Q"):
            cal = D[D.index < q.start_time]
            idx = D[(D.index >= q.start_time) & (D.index <= q.end_time)].index
            if len(idx) == 0:
                continue
            for k, (mk, grid) in makers.items():
                best = None
                for c in grid:
                    m = evaluate(mk(c, cal), cal.index)
                    loss = ratio * (m["eps"] - m["hits"]) + m["fas"]
                    if best is None or loss < best[0]:
                        best = (loss, c)
                m = evaluate(mk(best[1], cal), idx)
                for f in ("eps", "hits", "signals", "fas"):
                    res[k][f] += m[f]
                res[k]["leads"] += m["leads"]
            m = evaluate(D.pt.abs() > 0.005, idx)
            for f in ("eps", "hits", "signals", "fas"):
                res["BENCH 0.5%"][f] += m[f]
            res["BENCH 0.5%"]["leads"] += m["leads"]
        bench_loss = ratio * (res["BENCH 0.5%"]["eps"] - res["BENCH 0.5%"]["hits"]) + res["BENCH 0.5%"]["fas"]
        for k, a in res.items():
            loss = ratio * (a["eps"] - a["hits"]) + a["fas"]
            out.append(dict(ratio=ratio, rule=k, episodes=a["eps"], hits=a["hits"],
                            hit_rate=a["hits"] / max(1, a["eps"]), signals=a["signals"],
                            false_alarms=a["fas"], fa_rate=a["fas"] / max(1, a["signals"]),
                            median_lead=np.median(a["leads"]) if a["leads"] else np.nan,
                            expected_loss=loss, loss_vs_bench=loss / bench_loss - 1))
    T = pd.DataFrame(out)
    T.to_csv(RESULTS / "rq6_rules.csv", index=False)
    print(T.round(2).to_string(index=False))
    for _, r in T.iterrows():
        k = f"rq6.r{r.ratio}.{r.rule.split()[0]}"
        record(k + ".hit_rate", round(r.hit_rate, 2))
        record(k + ".signals", int(r.signals))
        record(k + ".loss_vs_bench", round(r.loss_vs_bench, 3))
