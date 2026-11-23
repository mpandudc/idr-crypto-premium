"""Settings shared by every stage of the pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"          # responses exactly as returned by each source
INTERIM = ROOT / "data" / "interim"  # aligned panels built from RAW
RESULTS = ROOT / "results"           # tables and summary numbers

# Sample window. END is exclusive; override with environment variables.
START = os.environ.get("TP_START", "2021-01-01")
END = os.environ.get("TP_END", "2026-11-07")

# Domestic assets used in the cross-asset analysis (all traded on Indodax against IDR
# and on Binance against USDT).
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "TRX", "AVAX", "BNB",
         "LTC", "DOT", "NEAR", "SUI", "TON", "PEPE", "SHIB"]

# Typical minutes to settle a deposit on the domestic platform (used as a control).
SETTLE_MIN = {"BTC": 60, "ETH": 5, "SOL": 1, "XRP": 1, "DOGE": 20, "ADA": 5, "LINK": 5,
              "TRX": 1, "AVAX": 1, "BNB": 1, "LTC": 30, "DOT": 1, "NEAR": 1, "SUI": 1,
              "TON": 1, "PEPE": 5, "SHIB": 5}

# One-minute sample months for price discovery: Jan, Apr, Jul, Oct of each year.
M1_MONTHS = [f"{y}-{m:02d}" for y in range(2021, 2027) for m in (1, 4, 7, 10)]

# Other domestic platforms with public history.
TOKO_RANGE = ("2024-01-01", "2025-12-15")
UPBIT_RANGE = ("2021-01-01", "2023-01-01")

# Event used for the access-suspension test (UTC, = 00:00 WIB 29 Oct 2026).
SUSPENSION_START = "2026-10-28 17:00"

# Episode definition for the decision rules: |total premium| above this for 2+ fixing days.
EPISODE_TH = 0.0075

USER_AGENT = "Mozilla/5.0 (MBA thesis research; public data)"
BP = 1e4
