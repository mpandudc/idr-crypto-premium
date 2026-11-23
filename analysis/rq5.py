"""RQ5 limits to arbitrage: band-TAR with bootstrap, tax regimes, equivalence test (TOST),
structural breaks, the October 2026 access suspension, and gaps across assets."""
import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from analysis.common import cost_band, half_life, record
from config import BP, COINS, RESULTS, SETTLE_MIN, SUSPENSION_START


def tar_fit(x, qs=np.arange(.05, .50, .05)):
    """Band threshold AR on first differences: dx = c + rhoL*(x-gL)- + rhoU*(x-gU)+ + e.
    Thresholds chosen by grid search over quantiles; supF against the linear AR."""
    x = pd.Series(x).dropna().values
    y, z = np.diff(x), x[:-1]
    n = len(y)
    lin = sm.OLS(y, sm.add_constant(z)).fit()
    best = None
    for ql in qs:
        for qu in 1 - qs:
            gl, gu = np.quantile(z, ql), np.quantile(z, qu)
            X = np.column_stack([np.ones(n), np.where(z < gl, z - gl, 0), np.where(z > gu, z - gu, 0)])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            ssr = ((y - X @ b) ** 2).sum()
            if best is None or ssr < best[0]:
                best = (ssr, gl, gu, b)
    F = (lin.ssr - best[0]) / (best[0] / (n - 3))
    return dict(gL=best[1], gU=best[2], rhoL=best[3][1], rhoU=best[3][2], supF=F, n=n), lin


def tar_test(x, reps=99, seed=1):
    """Residual bootstrap of supF under the linear AR null (Hansen-style)."""
    x = pd.Series(x).dropna().values
    r, lin = tar_fit(x)
    rng = np.random.default_rng(seed)
    e = lin.resid
    a, rho = lin.params
    cnt = 0
    for _ in range(reps):
        xs = np.empty(len(x))
        xs[0] = x[0]
        es = rng.choice(e, len(x) - 1)
        for t in range(1, len(x)):
            xs[t] = xs[t - 1] + a + rho * xs[t - 1] + es[t - 1]
        if tar_fit(xs, qs=np.arange(.1, .5, .1))[0]["supF"] >= r["supF"]:
            cnt += 1
    r["p_boot"] = (cnt + 1) / (reps + 1)
    return r


def breaks(x, max_k=7, min_size=60, jump=5):
    """Bai-Perron style mean breaks by dynamic programming, number chosen by BIC."""
    s = x.values * BP
    n = len(s)
    best = None
    for k in range(0, max_k + 1):
        bk = [n] if k == 0 else rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(s).predict(n_bkps=k)
        ssr = sum(((s[a:b] - s[a:b].mean()) ** 2).sum() for a, b in zip([0] + bk[:-1], bk))
        bic = n * np.log(ssr / n) + (2 * k + 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, bk)
    dates = [str(x.index[b - 1].date()) for b in best[2][:-1]]
    means = [round(float(s[a:b].mean()), 1) for a, b in zip([0] + best[2][:-1], best[2])]
    return best[1], dates, means


def run(H, D, J, A):
    # ---- band TAR
    rows = []
    for lab, x, reps in (("total_daily_2021_26", D.pt, 99), ("total_daily_2024_26", D.loc["2024":, "pt"], 99),
                         ("rupiah_daily_2021_26", D.pr, 99), ("crypto_hourly_2024_26", H.loc["2024":, "pc"], 49)):
        r = tar_test(x, reps=reps)
        r["series"] = lab
        rows.append(r)
        record(f"rq5.tar.{lab}.gL", round(r["gL"] * BP, 1))
        record(f"rq5.tar.{lab}.gU", round(r["gU"] * BP, 1))
        record(f"rq5.tar.{lab}.p_boot", round(r["p_boot"], 3))
        print("TAR", lab, round(r["gL"] * BP), round(r["gU"] * BP), "p", r["p_boot"], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "rq5_tar.csv", index=False)

    rows = []
    for lab, (a, b) in {"pre_tax": ("2021-01-01", "2022-04-30"), "pmk68": ("2022-05-01", "2025-07-31"),
                        "pmk50": ("2025-08-01", "2026-12-31")}.items():
        r, _ = tar_fit(D.loc[a:b, "pt"])
        r["regime"] = lab
        rows.append(r)
        record(f"rq5.regime.{lab}.gL", round(r["gL"] * BP, 1))
        record(f"rq5.regime.{lab}.gU", round(r["gU"] * BP, 1))
    pd.DataFrame(rows).to_csv(RESULTS / "rq5_tar_regimes.csv", index=False)

    # ---- TOST: is the crypto channel inside the professional cost band each year?
    rows = []
    for y in sorted(set(H.index.year)):
        x = H.loc[str(y), "pc"].dropna().resample("D").mean().dropna()
        if len(x) < 30:
            continue
        m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        mu, se = m.params[0], m.bse[0]
        up, lo = cost_band(pd.Timestamp(f"{y}-07-01"))
        p = max(1 - stats.norm.cdf((mu + lo) / se), stats.norm.cdf((mu - up) / se))
        rows.append(dict(year=y, mean_bp=mu * BP, se_bp=se * BP, band_lo=-lo * BP, band_up=up * BP, p_tost=p))
        record(f"rq5.tost.{y}.p", round(p, 5))
    pd.DataFrame(rows).to_csv(RESULTS / "rq5_tost.csv", index=False)

    # ---- structural breaks
    for lab, x in (("total", D.pt), ("rupiah", D.pr)):
        k, dates, means = breaks(x)
        record(f"rq5.breaks.{lab}.dates", dates)
        record(f"rq5.breaks.{lab}.means", means)
        print("breaks", lab, dates, means)

    # ---- access suspension of 29-31 Oct 2026 against placebo windows
    hc = H.pc

    def jump(t0):
        a = hc[t0 - pd.Timedelta(days=14):t0 - pd.Timedelta(hours=1)].mean()
        b = hc[t0:t0 + pd.Timedelta(hours=71)]
        return (b.mean() - a) if b.notna().sum() > 48 else np.nan

    t_ev = pd.Timestamp(SUSPENSION_START)
    if t_ev < H.index.max():
        obs = jump(t_ev)
        starts = pd.date_range(H.index.min() + pd.Timedelta(days=19), t_ev - pd.Timedelta(days=18), freq="3D")
        starts = starts + pd.Timedelta(hours=17 - starts[0].hour)
        pl = pd.Series([jump(t) for t in starts], index=starts).dropna()
        pl.to_csv(RESULTS / "rq5_suspension_placebos.csv")
        win = hc[t_ev:t_ev + pd.Timedelta(hours=71)]
        record("rq5.suspension.jump_bp", round(obs * BP, 1))
        record("rq5.suspension.window_mean_bp", round(win.mean() * BP, 1))
        record("rq5.suspension.window_max_bp", round(win.max() * BP, 1))
        record("rq5.suspension.placebos", len(pl))
        record("rq5.suspension.placebo_p99_bp", round(pl.quantile(.99) * BP, 1))
        record("rq5.suspension.placebo_max_bp", round(pl.max() * BP, 1))
        record("rq5.suspension.p", round((pl >= obs).sum() / (len(pl) + 1) + 1 / (len(pl) + 1), 4))
        print("suspension jump", round(obs * BP), "placebos", len(pl), "p99", round(pl.quantile(.99) * BP))

    # ---- gaps across assets (crypto channel only; all share the USDT/IDR leg)
    u = H.usdtidr
    rows = []
    for c in COINS:
        if c == "BTC":
            p, v, g = H.btcidr, H.btcv, H.glob
        else:
            p, v, g = A[f"{c}_p"], A[f"{c}_v"], A[f"{c}_g"]
        X = pd.DataFrame({"p": p, "v": v, "u": u, "g": g}).dropna()
        X = X[X.v > 0]
        X["pi"] = (np.log(X.p) - np.log(X.u) - np.log(X.g)) * BP
        for y, s in X.groupby(X.index.year):
            if len(s) < 500:
                continue
            rows.append(dict(coin=c, year=y, hours=len(s), mabs=s.pi.abs().median(),
                             in25=(s.pi.abs() < 25).mean(), hl=half_life(s.pi / BP, 200),
                             vol_bn=(s.v * s.p).median() / 1e9))
    R = pd.DataFrame(rows)
    R = R[R.vol_bn > 0].copy()
    R["settle"] = R.coin.map(SETTLE_MIN)
    R["lmabs"], R["lvol"], R["lset"] = np.log(R.mabs), np.log(R.vol_bn), np.log(R.settle)
    R.to_csv(RESULTS / "rq5_cross_asset.csv", index=False)
    grp = {"groups": R.coin.astype("category").cat.codes}
    m1 = smf.ols("lmabs ~ lvol + C(year)", R).fit(cov_type="cluster", cov_kwds=grp)
    m2 = smf.ols("lmabs ~ lvol + lset + C(year)", R).fit(cov_type="cluster", cov_kwds=grp)
    record("rq5.xasset.n", int(m1.nobs))
    record("rq5.xasset.elasticity", round(m1.params.lvol, 3))
    record("rq5.xasset.t", round(m1.tvalues.lvol, 1))
    record("rq5.xasset.r2", round(m1.rsquared, 2))
    record("rq5.xasset.t_settle", round(m2.tvalues.lset, 1))
    sp = {int(y): round(stats.spearmanr(g.mabs, g.vol_bn)[0], 2) for y, g in R.groupby("year")}
    record("rq5.xasset.spearman_by_year", sp)
    print("cross-asset n", int(m1.nobs), "elasticity", round(m1.params.lvol, 3), "t", round(m1.tvalues.lvol, 1),
          "R2", round(m1.rsquared, 2), "spearman", sp)
