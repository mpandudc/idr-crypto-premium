"""RQ3 price discovery in Bitcoin: one-minute VECM per eight-hour session, 24 sample months."""
import numpy as np
import pandas as pd

from analysis.common import record, vecm_shares
from config import INTERIM, M1_MONTHS, RESULTS


def run():
    out = []
    for mo in M1_MONTHS:
        f = INTERIM / "m1" / f"{mo}.csv"
        if not f.exists():
            continue
        a = pd.Timestamp(mo + "-01")
        b = a + pd.offsets.MonthBegin(1)
        m = pd.read_csv(f, index_col=0, parse_dates=True)
        idx = pd.date_range(a, b, freq="1min", inclusive="left")
        X = pd.DataFrame({"bi": m.bi.reindex(idx).ffill(limit=5), "ui": m.ui.reindex(idx).ffill(limit=30),
                          "g": m.g.reindex(idx), "traded": m.bv.reindex(idx).fillna(0) > 0}).dropna()
        X["dom"] = np.log(X.bi / X.ui)  # domestic BTC price in USDT terms
        X["glob"] = np.log(X.g)
        wib = X.index + pd.Timedelta(hours=7)
        X["sess"] = np.select([(wib.hour >= 7) & (wib.hour < 15), (wib.hour >= 15) & (wib.hour < 23)],
                              ["Asia", "Europe"], "US")
        X["day"] = wib.normalize()
        X["weekend"] = wib.dayofweek >= 5
        for (day, s), x in X.groupby(["day", "sess"]):
            if len(x) < 300:
                continue
            r = vecm_shares(x[["dom", "glob"]].values, k=5)
            if r:
                r.update(month=mo, year=int(mo[:4]), sess=s, weekend=bool(x.weekend.iloc[0]),
                         traded=x.traded.mean())
                out.append(r)
        print("  rq3", mo, len(X), flush=True)
    R = pd.DataFrame(out)
    R.to_csv(RESULTS / "rq3_price_discovery.csv", index=False)
    cols = ["is_lo", "is_hi", "cs", "ils"]
    by_year = R.groupby("year")[cols].median().assign(n=R.groupby("year").size())
    print(by_year.round(3).to_string())
    print(R.groupby("sess")[cols].median().round(3).to_string())
    record("rq3.sessions_admissible", int(len(R)))
    for y, r in by_year.iterrows():
        record(f"rq3.{y}.ils_median", round(r.ils, 3))
    record("rq3.share_ils_below_half", round((R.ils < 0.5).mean(), 3))
    for s, v in R.groupby("sess").ils.median().items():
        record(f"rq3.sess.{s}.ils_median", round(v, 3))
    record("rq3.weekend.ils_median", round(R[R.weekend].ils.median(), 3))
