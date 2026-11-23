"""Gaps between domestic platforms (Tokocrypto 2024-25, Upbit Indonesia 2021-22) and the
net-taker-flow proxy for the rupiah channel."""
import numpy as np
import pandas as pd

from analysis.common import record
from config import BP, RESULTS


def _hac(y, x, L=5):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    S0 = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S0 += w * (G + G.T)
    se = np.sqrt(np.diag(XtX @ S0 @ XtX))
    return b[1], b[1] / se[1], 1 - e.var() / y.var(), len(y)


def run(H, D, J, V):
    H = H.copy()
    H["u"] = H.u.fillna(1.0)

    def pair(btc, usdt, vb, vu):
        X = pd.DataFrame({"b2": btc, "u2": usdt, "vb": vb, "vu": vu}).join(
            H[["btcidr", "usdtidr", "btcv", "usdtv", "glob", "u"]], how="inner")
        X = X[(X.vb > 0) & (X.vu > 0) & (X.btcv > 0) & (X.usdtv > 0)].dropna()
        X["d_btc"] = BP * np.log(X.b2 / X.btcidr)
        X["d_usdt"] = BP * np.log(X.u2 / X.usdtidr)
        return X

    venues = {"toko": pair(V["toko_BTC"].c, V["toko_USDT"].c, V["toko_BTC"].v, V["toko_USDT"].v),
              "upbit": pair(V["upbit_BTC"].c, V["upbit_USDT"].c, V["upbit_BTC"].qv, V["upbit_USDT"].qv)}
    rows = []
    for name, X in venues.items():
        Dv = X[(X.index.hour == 2) & (X.index.dayofweek < 5)].copy()
        Dv.index = Dv.index.normalize()
        Dv = Dv.join(J.rename("jis"), how="inner")
        S = Dv.u * Dv.jis
        tot_i, tot_o = BP * np.log(Dv.btcidr / (Dv.glob * S)), BP * np.log(Dv.b2 / (Dv.glob * S))
        rup_i, rup_o = BP * np.log(Dv.usdtidr / S), BP * np.log(Dv.u2 / S)
        for y in sorted(set(X.index.year)):
            h = X[X.index.year == y]
            m = Dv.index.year == y
            r = dict(venue=name, year=y, hours=len(h), days=int(m.sum()),
                     gap_btc_medabs=h.d_btc.abs().median(), gap_usdt_medabs=h.d_usdt.abs().median())
            if m.sum() > 20:
                r.update(total_indodax=tot_i[m].mean(), total_other=tot_o[m].mean(),
                         corr_total=np.corrcoef(tot_i[m], tot_o[m])[0, 1],
                         rupiah_indodax=rup_i[m].mean(), rupiah_other=rup_o[m].mean(),
                         corr_rupiah=np.corrcoef(rup_i[m], rup_o[m])[0, 1],
                         rupiah_share_indodax=np.cov(rup_i[m], tot_i[m])[0, 1] / tot_i[m].var(),
                         rupiah_share_other=np.cov(rup_o[m], tot_o[m])[0, 1] / tot_o[m].var())
            rows.append(r)
            for k, v in r.items():
                if k not in ("venue", "year"):
                    record(f"xv.{name}.{y}.{k}", round(float(v), 3))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "xvenue.csv", index=False)
    print(T.round(2).to_string(index=False))

    # valve proxy: previous day's net taker buying of USDT/IDR on Tokocrypto
    tu = V["toko_USDT"]
    tu = tu[tu.v > 0]
    net = ((2 * tu.takerbuy - tu.v) * tu.c).resample("D").sum()
    vol = (tu.v * tu.c).resample("D").sum()
    imb = net / vol
    X = venues["toko"]
    Dv = X[X.index.hour == 2].copy()
    Dv.index = Dv.index.normalize()
    Dv = Dv.join(J.rename("jis"), how="inner")
    Z = pd.DataFrame({"imb_prev": imb.shift(1), "rup": BP * np.log(Dv.u2 / (Dv.u * Dv.jis))}).dropna()
    Z["drup"] = Z.rup.diff()
    Z = Z.dropna()
    zs = (Z.imb_prev - Z.imb_prev.mean()) / Z.imb_prev.std()
    b, t, r2, n = _hac(Z.drup.values, zs.values)
    bl, tl, _, _ = _hac(Z.rup.values, zs.values)
    record("xv.valve.change_per_sd_bp", round(b, 2))
    record("xv.valve.t", round(t, 2))
    record("xv.valve.r2", round(r2, 3))
    record("xv.valve.n", int(n))
    record("xv.valve.level_t", round(tl, 2))
    print("valve: +1 sd net taker buying ->", round(b, 2), "bp next-morning change, t", round(t, 2), "R2", round(r2, 3), "n", n)
