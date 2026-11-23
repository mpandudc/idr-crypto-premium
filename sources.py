"""Downloaders for every public source, with a file cache of raw responses.

Each request covers a fixed time window. A window that has fully closed is written to
data/raw/<source>/<series>/<window>.json and never requested again, so a rerun only
downloads what is new. Windows that are still open are fetched but not cached.
"""
import json
import time
from io import StringIO

import pandas as pd
import requests

from config import RAW, USER_AGENT

S = requests.Session()
S.headers["User-Agent"] = USER_AGENT


def _get(url, params, pause=0.0, tries=8):
    last = None
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 429:  # rate limited: wait and retry
                time.sleep(30 * (i + 1))
                continue
            r.raise_for_status()
            if pause:
                time.sleep(pause)
            return r.json()
        except Exception as e:  # network error: back off and retry
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url} {params} ({last})")


def _closed(t_end):
    t = pd.Timestamp(t_end)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)


def _cached(path, fetch, t_end):
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    if _closed(t_end):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def _ts(x):
    return pd.Timestamp(x, tz="UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x)


# ---------------------------------------------------------------- Indodax
def indodax(symbol, start, end, tf="60"):
    """Indodax TradingView history. tf '60' = hourly, '1' = one minute.
    Returns close (c) and base-asset volume (v) indexed by bar open time (UTC)."""
    step = {"60": 25 * 86400, "1": 8 * 86400}[tf]
    t, te = int(_ts(start).timestamp()), int(_ts(end).timestamp())
    rows = []
    while t < te:
        t1 = min(t + step, te)
        p = RAW / "indodax" / f"{symbol}_{tf}" / f"{t}_{t1}.json"
        d = _cached(p, lambda: _get("https://indodax.com/tradingview/history_v2",
                                    {"symbol": symbol, "tf": tf, "from": t, "to": t1}, pause=2.5),
                    pd.Timestamp(t1, unit="s"))
        rows += [(x["Time"], float(x["Close"]), float(x["Volume"])) for x in (d or [])]
        t = t1
    df = pd.DataFrame(rows, columns=["ts", "c", "v"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    return df[(df.index >= _ts(start)) & (df.index < _ts(end))]


# ---------------------------------------------------------------- Binance
def binance(symbol, start, end, interval="1h"):
    """Binance public klines (data-api.binance.vision). Close and quote volume (USDT)."""
    ms = {"1h": 3_600_000, "1m": 60_000}[interval]
    t, te = int(_ts(start).timestamp() * 1000), int(_ts(end).timestamp() * 1000)
    rows = []
    while t < te:
        t1 = min(t + 1000 * ms, te)
        p = RAW / "binance" / f"{symbol}_{interval}" / f"{t}_{t1}.json"
        d = _cached(p, lambda: _get("https://data-api.binance.vision/api/v3/klines",
                                    {"symbol": symbol, "interval": interval, "startTime": t,
                                     "endTime": t1 - 1, "limit": 1000}, pause=0.1),
                    pd.Timestamp(t1, unit="ms"))
        rows += [(x[0], float(x[4]), float(x[7])) for x in (d or [])]
        t = t1
    df = pd.DataFrame(rows, columns=["ts", "c", "qv"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df.set_index("ts").sort_index()


# ---------------------------------------------------------------- Coinbase
def coinbase_usdt(start, end):
    """Hourly USDT/USD close from Coinbase Exchange (history begins in mid-2021)."""
    t0, te = _ts(start), _ts(end)
    rows = []
    while t0 < te:
        t1 = min(t0 + pd.Timedelta(hours=300), te)
        p = RAW / "coinbase" / "USDT-USD_3600" / f"{int(t0.timestamp())}_{int(t1.timestamp())}.json"
        d = _cached(p, lambda: _get("https://api.exchange.coinbase.com/products/USDT-USD/candles",
                                    {"granularity": 3600, "start": t0.isoformat(),
                                     "end": t1.isoformat()}, pause=0.3), t1)
        rows += [(x[0], float(x[4])) for x in (d or []) if isinstance(x, list)]
        t0 = t1
    df = pd.DataFrame(rows, columns=["ts", "u"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True)
    return df.set_index("ts").sort_index()


# ---------------------------------------------------------------- JISDOR (Bank Indonesia)
_BI = "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx"
_BI_PREFIX = "ctl00$ctl51$g_8e393016_c37f_48c2_9aa2_ab195a9c1128$ctl00$"
_MONTHS = {"Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6, "Juli": 7,
           "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12}


def _bi_form(html, f, t):
    from bs4 import BeautifulSoup
    form = BeautifulSoup(html, "lxml").find("form")
    d = {}
    for i in form.find_all("input"):
        n = i.get("name")
        if not n or (i.get("type") or "text").lower() in ("submit", "button", "image", "checkbox", "radio"):
            continue
        d[n] = i.get("value", "")
    for k in list(d):
        if k.startswith(_BI_PREFIX) and ("DateFrom" in k or "DateTo" in k or "TextBoxFrom" in k):
            d[k] = f if "From" in k else t
    return d


def _bi_scrape(f, t):
    """Scrape the JISDOR table page by page (ASP.NET postbacks) for dates f..t (YYYY-MM-DD)."""
    from bs4 import BeautifulSoup
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"
    r = s.get(_BI, timeout=60)
    d = _bi_form(r.text, f, t)
    d[_BI_PREFIX + "ButtonCari"] = "Cari"
    r = s.post(_BI, data=d, timeout=90)
    pages = []
    for _ in range(300):
        pages.append(pd.read_html(StringIO(r.text))[0])
        nxt = BeautifulSoup(r.text, "lxml").find("input", {"class": "next"})
        if nxt is None or "aspNetDisabled" in (nxt.get("class") or []) or nxt.get("disabled"):
            break
        d = _bi_form(r.text, f, t)
        d[nxt["name"] + ".x"], d[nxt["name"] + ".y"] = "5", "5"
        for a in range(4):
            try:
                r = s.post(_BI, data=d, timeout=90)
                break
            except Exception:
                time.sleep(5 * (a + 1))
        time.sleep(1)
    return pd.concat(pages).drop_duplicates()


def jisdor(start, end):
    """JISDOR reference rate (IDR per USD), one value per fixing day. Raw tables are cached per year."""
    out = []
    for y in range(_ts(start).year, _ts(end).year + 1):
        a, b = max(_ts(start), _ts(f"{y}-01-01")), min(_ts(end), _ts(f"{y}-12-31"))
        p = RAW / "jisdor" / f"{y}.csv"
        if p.exists():
            raw = pd.read_csv(p)
        else:
            raw = _bi_scrape(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
            if _closed(_ts(f"{y}-12-31") + pd.Timedelta(days=1)):
                p.parent.mkdir(parents=True, exist_ok=True)
                raw.to_csv(p, index=False)
        out.append(raw)
    j = pd.concat(out)

    def day(s):
        d, m, y = s.split()
        return pd.Timestamp(int(y), _MONTHS[m], int(d))

    j["date"] = j.Tanggal.map(day)
    j["jisdor"] = (j.Kurs.str.replace("Rp", "").str.replace(".", "", regex=False)
                   .str.replace(",", ".").str.strip().astype(float))
    j = j.drop_duplicates("date").set_index("date").sort_index().jisdor
    return j[(j.index >= _ts(start).tz_localize(None)) & (j.index < _ts(end).tz_localize(None))]


# ---------------------------------------------------------------- Yahoo Finance USD/IDR
def yahoo_usdidr():
    """Hourly market USD/IDR quote (IDR=X). Yahoo serves only the last 730 days, so each
    download is merged into data/raw/yahoo/IDR=X_60m.csv. Used only as a competing predictor."""
    import yfinance as yf
    p = RAW / "yahoo" / "IDR=X_60m.csv"
    old = pd.read_csv(p, index_col=0, parse_dates=True).fx if p.exists() else pd.Series(dtype=float)
    y = yf.download("IDR=X", period="730d", interval="60m", progress=False, auto_adjust=False)
    if isinstance(y.columns, pd.MultiIndex):
        y.columns = y.columns.get_level_values(0)
    new = y.Close.rename("fx")
    new.index = pd.to_datetime(new.index, utc=True)
    if len(old):
        old.index = pd.to_datetime(old.index, utc=True)
    fx = pd.concat([old, new]).groupby(level=0).last().rename("fx")
    p.parent.mkdir(parents=True, exist_ok=True)
    fx.to_frame().to_csv(p)
    return fx


# ---------------------------------------------------------------- Tokocrypto (legacy public API)
def tokocrypto(symbol, start, end):
    """Hourly klines with taker-buy volume (Binance layout: field 9 = taker buy base volume)."""
    t, te = int(_ts(start).timestamp() * 1000), int(_ts(end).timestamp() * 1000)
    rows = []
    while t < te:
        t1 = min(t + 1000 * 3_600_000, te)
        p = RAW / "tokocrypto" / f"{symbol}_1h" / f"{t}_{t1}.json"
        d = _cached(p, lambda: _get("https://cloudme-toko.2meta.app/api/v1/klines",
                                    {"symbol": symbol, "interval": "1h", "limit": 1000,
                                     "startTime": t, "endTime": t1 - 1}, pause=0.2),
                    pd.Timestamp(t1, unit="ms"))
        rows += [(x[0], float(x[4]), float(x[5]), float(x[9]), float(x[7]), int(x[8])) for x in (d or [])]
        t = t1
    df = pd.DataFrame(rows, columns=["ts", "c", "v", "takerbuy", "qv", "n"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df.set_index("ts").sort_index()


# ---------------------------------------------------------------- Upbit Indonesia
def upbit(market, start, end):
    """Hourly candles; each request returns up to 200 candles ending before 'to'."""
    t0, te = _ts(start), _ts(end)
    rows = []
    while t0 < te:
        t1 = min(t0 + pd.Timedelta(hours=200), te)
        p = RAW / "upbit" / f"{market}_60" / f"{int(t0.timestamp())}_{int(t1.timestamp())}.json"
        d = _cached(p, lambda: _get("https://id-api.upbit.com/v1/candles/minutes/60",
                                    {"market": market, "count": 200,
                                     "to": t1.strftime("%Y-%m-%dT%H:%M:%SZ")}, pause=0.15), t1)
        rows += [(x["candle_date_time_utc"], float(x["trade_price"]), float(x["candle_acc_trade_price"]))
                 for x in (d or []) if isinstance(x, dict)]
        t0 = t1
    df = pd.DataFrame(rows, columns=["ts", "c", "qv"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    df = df.set_index("ts").sort_index()
    return df[(df.index >= _ts(start)) & (df.index < _ts(end))]
