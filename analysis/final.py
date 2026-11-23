"""Full estimators that replace the simplified pilot versions (Section III.4).

  tar     band-TAR with a fine threshold grid and 999 bootstrap draws, plus bootstrap
          percentile intervals for the two thresholds
  rules   rolling-origin evaluation with expanding and two-year windows, and the
          rule set frozen on all data available (dry run of the pre-analysis plan freeze)
"""
import json

import numpy as np
import pandas as pd

from analysis.common import cost_band, episodes, half_life, record
from analysis.rq5 import tar_fit
from config import BP, EPISODE_TH, RESULTS

FINE = np.round(np.arange(0.05, 0.451, 0.025), 3)


def tar_full(x, reps=999, seed=7, grid=FINE):
    """supF test against a linear AR(1) by residual bootstrap, and threshold intervals
    from a bootstrap of the fitted TAR (draws from its own residuals)."""
    x = pd.Series(x).dropna().values
    r, lin = tar_fit(x, qs=grid)
    rng = np.random.default_rng(seed)
    e, (a, rho) = lin.resid, lin.params
    exceed = 0
    for _ in range(reps):
        xs = np.empty(len(x)); xs[0] = x[0]
        es = rng.choice(e, len(x) - 1)
        for t in range(1, len(x)):
            xs[t] = xs[t - 1] + a + rho * xs[t - 1] + es[t - 1]
        if tar_fit(xs, qs=grid)[0]["supF"] >= r["supF"]:
            exceed += 1
    r["p_boot"] = (exceed + 1) / (reps + 1)
    # threshold intervals: simulate from the fitted TAR
    y, z = np.diff(x), x[:-1]
    X = np.column_stack([np.ones(len(z)), np.where(z < r["gL"], z - r["gL"], 0), np.where(z > r["gU"], z - r["gU"], 0)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    gl, gu = [], []
    for _ in range(min(reps, 299)):
        xs = np.empty(len(x)); xs[0] = x[0]
        es = rng.choice(res, len(x) - 1)
        for t in range(1, len(x)):
            p = xs[t - 1]
            xs[t] = p + b[0] + b[1] * min(p - r["gL"], 0) + b[2] * max(p - r["gU"], 0) + es[t - 1]
        f = tar_fit(xs, qs=grid)[0]
        gl.append(f["gL"]); gu.append(f["gU"])
    r["gL_lo"], r["gL_hi"] = np.percentile(gl, [5, 95])
    r["gU_lo"], r["gU_hi"] = np.percentile(gu, [5, 95])
    return r


def run_tar(H, D):
    rows = []
    series = [("total_daily_2021_26", D.pt), ("total_daily_2024_26", D.loc["2024":, "pt"]),
              ("rupiah_daily_2021_26", D.pr),
              ("total_pre_tax", D.loc[:"2022-04-30", "pt"]), ("total_pmk68", D.loc["2022-05-01":"2025-07-31", "pt"]),
              ("total_pmk50", D.loc["2025-08-01":, "pt"])]
    for lab, x in series:
        r = tar_full(x)
        r["series"] = lab
        rows.append(r)
        for k in ("gL", "gU", "gL_lo", "gL_hi", "gU_lo", "gU_hi"):
            record(f"final.tar.{lab}.{k}", round(r[k] * BP, 1))
        record(f"final.tar.{lab}.p_boot", round(r["p_boot"], 3))
        record(f"final.tar.{lab}.rhoL", round(r["rhoL"], 3))
        record(f"final.tar.{lab}.rhoU", round(r["rhoU"], 3))
        print(lab, "band", round(r["gL"] * BP), round(r["gU"] * BP),
              "90% CI gL", [round(v * BP) for v in (r["gL_lo"], r["gL_hi"])],
              "gU", [round(v * BP) for v in (r["gU_lo"], r["gU_hi"])],
              "rhoL", round(r["rhoL"], 2), "rhoU", round(r["rhoU"], 2), "p", r["p_boot"], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "final_tar.csv", index=False)


# ------------------------------------------------------------------ decision rules
LEAD = 5
GRIDS = {
    "L1 level": [0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
    "L2 level and rising": [0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006],
    "L3 rupiah channel level": [0.003, 0.004, 0.005, 0.006, 0.007],
    "L4 overnight USDT/IDR vs last fixing": [0.003, 0.004, 0.005, 0.006, 0.008],
}


def run_rules(H, D):
    D = D.copy()
    u9 = H.usdtidr[H.index.hour == 1]
    u9.index = u9.index.normalize()
    D["a7"] = np.log(u9.reindex(D.index)) - np.log(D.jis.shift(1))
    D["dpt"] = D.pt.diff()
    ep, inep = episodes(D.pt, EPISODE_TH)
    make = {
        "L1 level": lambda c: D.pt.abs() > c,
        "L2 level and rising": lambda c: (D.pt.abs() > c) & (np.sign(D.pt) * D.dpt > 0),
        "L3 rupiah channel level": lambda c: D.pr.abs() > c,
        "L4 overnight USDT/IDR vs last fixing": lambda c: D.a7.abs() > c,
    }

    def evaluate(sig, idx):
        s = sig.reindex(idx).fillna(False).astype(bool).values
        e, ie = ep.reindex(idx).values, inep.reindex(idx).values
        starts = np.where(e == 1)[0]
        hits = sum(1 for k in starts if s[max(0, k - LEAD):k].any())
        sp = np.where(s & ~ie)[0]
        fas = sum(0 if (e[j + 1:j + 1 + LEAD] == 1).any() else 1 for j in sp)
        return len(starts), hits, len(sp), fas

    def choose(rule, idx, ratio, min_hit=0.5):
        """Threshold with the lowest expected loss among those that catch at least
        min_hit of the calibration episodes (falls back to the loss minimum)."""
        best, fallback = None, None
        for c in GRIDS[rule]:
            n, h, s, f = evaluate(make[rule](c), idx)
            loss = ratio * (n - h) + f
            if fallback is None or loss < fallback[0]:
                fallback = (loss, c)
            if n and h / n >= min_hit and (best is None or loss < best[0]):
                best = (loss, c)
        return (best or fallback)[1]

    out = []
    for window in ("expanding", "two-year"):
        for ratio in (3, 10):
            for rule in list(make) + ["BENCH 0.5%"]:
                tot = np.zeros(4)
                for q in pd.period_range("2022Q1", "2026Q4", freq="Q"):
                    lo = pd.Timestamp("2000-01-01") if window == "expanding" else q.start_time - pd.DateOffset(years=2)
                    cal = D[(D.index < q.start_time) & (D.index >= lo)].index
                    idx = D[(D.index >= q.start_time) & (D.index <= q.end_time)].index
                    if len(idx) == 0:
                        continue
                    sig = D.pt.abs() > 0.005 if rule.startswith("BENCH") else make[rule](choose(rule, cal, ratio))
                    tot += evaluate(sig, idx)
                n, h, s, f = tot
                out.append(dict(window=window, ratio=ratio, rule=rule, episodes=int(n), hits=int(h), hit_rate=h / n,
                                signals=int(s), false_alarms=int(f), expected_loss=ratio * (n - h) + f))
    T = pd.DataFrame(out)
    for (w, r), g in T.groupby(["window", "ratio"]):
        bench = g.loc[g.rule == "BENCH 0.5%", "expected_loss"].iloc[0]
        T.loc[g.index, "loss_vs_bench"] = g.expected_loss / bench - 1
    T.to_csv(RESULTS / "final_rules.csv", index=False)
    print(T.round(2).to_string(index=False))
    for _, r in T.iterrows():
        k = f"final.rules.{r.window}.r{r.ratio}.{r.rule.split()[0]}"
        record(k + ".hit_rate", round(r.hit_rate, 2))
        record(k + ".signals", int(r.signals))
        record(k + ".loss_vs_bench", round(r.loss_vs_bench, 3))

    # dry-run freeze on all data available (cost ratio 10, hit-rate floor 0.5)
    frozen = {"calibration_end": str(D.index.max().date()), "cost_ratio": 10, "min_hit_rate": 0.5,
              "episode_threshold": EPISODE_TH, "lead_days": LEAD,
              "half_life_days": round(float(half_life(D.pt)), 2), "rules": {}}
    for rule in make:
        frozen["rules"][rule] = choose(rule, D.index, 10)
    (RESULTS / "frozen_rules_dryrun.json").write_text(json.dumps(frozen, indent=1))
    print("frozen (dry run):", frozen)


# ------------------------------------------------------------------ direction-specific cost band
REGIMES = {"total_pre_tax": ("2021-01-01", "2022-04-30"), "total_pmk68": ("2022-05-01", "2025-07-31"),
           "total_pmk50": ("2025-08-01", "2099-12-31")}


def run_band(H, D):
    """Compare the estimated thresholds of each tax regime with the crypto-side cost of closing the
    total premium in each direction (common.cost_band, kind="total").

    In a band model around an equilibrium m, gU = m + cU and gL = m - cL, so the band width gU - gL
    estimates the round trip cU + cL, and the asymmetry (gU - m) - (m - gL) = gU + gL - 2m estimates
    cU - cL. m is taken as the regime median of the premium. The rupiah conversion is not observed, so
    width minus the crypto-side round trip is the implied rupiah-side cost."""
    T = pd.read_csv(RESULTS / "final_tar.csv").set_index("series")
    rows = []
    for lab, (a, b) in REGIMES.items():
        x = D.loc[a:b, "pt"]
        cu, cl = cost_band(pd.Timestamp(a), kind="total")
        r = T.loc[lab]
        m = x.median()
        rows.append(dict(regime=lab, days=len(x), p_boot=r.p_boot, gL=r.gL * BP, gU=r.gU * BP,
                         gL_lo=r.gL_lo * BP, gL_hi=r.gL_hi * BP, gU_lo=r.gU_lo * BP, gU_hi=r.gU_hi * BP,
                         median=m * BP, cost_up=cu * BP, cost_lo=cl * BP,
                         width=(r.gU - r.gL) * BP, cost_round_trip=(cu + cl) * BP,
                         implied_rupiah_cost=(r.gU - r.gL - cu - cl) * BP,
                         asym=(r.gU + r.gL - 2 * m) * BP, cost_asym=(cu - cl) * BP,
                         share_above_up=float((x - m > cu).mean()), share_below_lo=float((m - x > cl).mean())))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "final_band.csv", index=False)
    print(R.round(2).to_string(index=False))
    for _, r in R.iterrows():
        k = "final.band." + r.regime.replace("total_", "")
        for c in ("cost_up", "cost_lo", "width", "cost_round_trip", "implied_rupiah_cost", "asym", "cost_asym", "median"):
            record(f"{k}.{c}", round(float(r[c]), 1))
    a68, a50 = R.set_index("regime").loc[["total_pmk68", "total_pmk50"], ["asym", "cost_asym"]].values
    record("final.band.asym_change_observed", round(float(a50[0] - a68[0]), 1))
    record("final.band.asym_change_predicted", round(float(a50[1] - a68[1]), 1))
