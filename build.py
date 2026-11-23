"""Stage 1 and 2: download (through the cache) and align every series into analysis panels.

Outputs in data/interim/:
  hourly_master.csv   BTC, USDT and ETH on Indodax, BTC and ETH on Binance, USDT/USD, USD/IDR quote
  daily_panel.csv     10:00 WIB fixing-time panel with the premium and its two channels
  alt_hourly.csv      price, volume and global price for each asset in config.COINS
  m1/YYYY-MM.csv      one-minute BTC/IDR, USDT/IDR (Indodax) and BTC/USDT (Binance) sample months
  xvenue_toko.csv, xvenue_upbit.csv   other domestic platforms
  jisdor.csv
"""
import numpy as np
import pandas as pd

import sources as src
from config import BP, COINS, END, INTERIM, M1_MONTHS, START, TOKO_RANGE, UPBIT_RANGE


def _naive(x):
    x = x.copy()
    x.index = x.index.tz_convert(None)
    return x


def build_jisdor():
    # One year of history before START so the first fixing-to-fixing change exists.
    j = src.jisdor(str(pd.Timestamp(START).year - 1) + "-01-01", END)
    j.rename("jisdor").to_csv(INTERIM / "jisdor.csv")
    return j


def build_hourly():
    bi = src.indodax("BTCIDR", START, END)
    ui = src.indodax("USDTIDR", START, END)
    ei = src.indodax("ETHIDR", START, END)
    g = src.binance("BTCUSDT", START, END)
    ge = src.binance("ETHUSDT", START, END)
    u = src.coinbase_usdt(START, END)
    try:
        fx = src.yahoo_usdidr()
    except Exception as e:  # optional series
        print("Yahoo USD/IDR unavailable:", e)
        fx = pd.Series(dtype=float, name="fx")
    H = pd.DataFrame({"btcidr": bi.c, "btcv": bi.v, "usdtidr": ui.c, "usdtv": ui.v,
                      "ethidr": ei.c, "ethv": ei.v, "glob": g.c, "globqv": g.qv, "eglob": ge.c})
    H = H.join(u).join(fx.rename("fx"))
    H.index.name = "ts"
    H = _naive(H)
    H.to_csv(INTERIM / "hourly_master.csv")
    return H


def daily_panel(H, J):
    """10:00 WIB (02:00 UTC) weekday observations joined to the JISDOR fixing of the same day.
    pc = crypto channel, pr = rupiah channel, pt = total premium (all in log points)."""
    H = H.copy()
    H["u"] = H.u.ffill(limit=24).fillna(1.0)  # USDT at parity before Coinbase history starts
    H = H.dropna(subset=["btcidr", "usdtidr", "glob"])
    H["pc"] = np.log(H.btcidr) - np.log(H.usdtidr) - np.log(H.glob)
    H.loc[H.btcv <= 0, "pc"] = np.nan  # no trade in the hour: price is stale
    H["pce"] = np.log(H.ethidr) - np.log(H.usdtidr) - np.log(H.eglob)
    H.loc[H.ethv <= 0, "pce"] = np.nan
    D = H[(H.index.hour == 2) & (H.index.dayofweek < 5)].copy()
    D.index = D.index.normalize()
    D = D.join(J.rename("jis"), how="inner")
    D["pr"] = np.log(D.usdtidr) - np.log(D.u * D.jis)
    D["pt"] = D.pc + D.pr
    D["pte"] = D.pce + D.pr
    D = D.dropna(subset=["pt"])
    D = D[D.index >= START]
    rv = np.log(H.glob).diff().groupby(H.index.normalize()).std() * np.sqrt(24)
    D["rv"] = rv.reindex(D.index)
    D["ret90"] = np.log(D.glob).diff(63)
    D.index.name = "date"
    return D, H


def build_alts():
    cols = {}
    for c in COINS:
        if c == "BTC":
            continue
        di = src.indodax(c + "IDR", START, END)
        gb = src.binance(c + "USDT", START, END)
        cols.update({f"{c}_p": di.c, f"{c}_v": di.v, f"{c}_g": gb.c, f"{c}_gqv": gb.qv})
        print("  alt", c, len(di), len(gb), flush=True)
    A = _naive(pd.DataFrame(cols))
    A.index.name = "ts"
    A.to_csv(INTERIM / "alt_hourly.csv")
    return A


def build_m1():
    (INTERIM / "m1").mkdir(parents=True, exist_ok=True)
    for mo in M1_MONTHS:
        a = pd.Timestamp(mo + "-01")
        b = a + pd.offsets.MonthBegin(1)
        if b > pd.Timestamp(END):
            continue
        bi = src.indodax("BTCIDR", a, b, tf="1")
        ui = src.indodax("USDTIDR", a, b, tf="1")
        g = src.binance("BTCUSDT", a, b, interval="1m")
        M = _naive(pd.DataFrame({"bi": bi.c, "bv": bi.v, "ui": ui.c, "g": g.c}))
        M.index.name = "ts"
        M.to_csv(INTERIM / "m1" / f"{mo}.csv")
        print("  m1", mo, len(bi), len(ui), len(g), flush=True)


def build_xvenue():
    for sym in ("BTCIDR", "USDTIDR"):
        t = _naive(src.tokocrypto(sym, *TOKO_RANGE))
        t.add_prefix(sym + "_").to_csv(INTERIM / f"xvenue_toko_{sym}.csv")
    for m in ("IDR-BTC", "IDR-USDT"):
        u = _naive(src.upbit(m, *UPBIT_RANGE))
        u.add_prefix(m + "_").to_csv(INTERIM / f"xvenue_upbit_{m}.csv")


def run(parts=("jisdor", "hourly", "alts", "m1", "xvenue")):
    INTERIM.mkdir(parents=True, exist_ok=True)
    if "jisdor" in parts:
        print("JISDOR", flush=True)
        build_jisdor()
    if "hourly" in parts:
        print("hourly master", flush=True)
        build_hourly()
    if "jisdor" in parts or "hourly" in parts:
        H = load_hourly()
        J = load_jisdor()
        D, _ = daily_panel(H, J)
        D.to_csv(INTERIM / "daily_panel.csv")
        print("  daily panel", len(D), D.index.min().date(), D.index.max().date(), flush=True)
    if "alts" in parts:
        print("alts", flush=True)
        build_alts()
    if "m1" in parts:
        print("one-minute samples", flush=True)
        build_m1()
    if "xvenue" in parts:
        print("other platforms", flush=True)
        build_xvenue()


# ---------------------------------------------------------------- loaders used by the analyses
def load_hourly():
    return pd.read_csv(INTERIM / "hourly_master.csv", index_col=0, parse_dates=True)


def load_jisdor():
    return pd.read_csv(INTERIM / "jisdor.csv", index_col=0, parse_dates=True).iloc[:, 0]


def load_panels():
    """Hourly frame with channels and the daily fixing panel, as used by RQ1 to RQ6."""
    D, H = daily_panel(load_hourly(), load_jisdor())
    return H, D, load_jisdor()


def load_alts():
    return pd.read_csv(INTERIM / "alt_hourly.csv", index_col=0, parse_dates=True)


def load_xvenue():
    out = {}
    for k, f in (("toko_BTC", "xvenue_toko_BTCIDR"), ("toko_USDT", "xvenue_toko_USDTIDR"),
                 ("upbit_BTC", "xvenue_upbit_IDR-BTC"), ("upbit_USDT", "xvenue_upbit_IDR-USDT")):
        d = pd.read_csv(INTERIM / f"{f}.csv", index_col=0, parse_dates=True)
        d.columns = [c.split("_", 1)[1] for c in d.columns]
        out[k] = d
    return out
