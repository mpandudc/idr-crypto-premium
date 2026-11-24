# Crypto price premium in Indonesia's rupiah market: replication code

Code for the MBA project thesis "The Crypto Price Premium in Indonesia's Rupiah Market: Price Discovery,
Limits to Arbitrage, and Decision Rules for Crypto Bourses" (Muhammad Pandu Dwi Cahyo, SBM ITB).

The pipeline downloads every series from public sources, builds the panels and reproduces each number
in the results chapter. `run.py verify` compares the output with `expected.json`.

No data is included. Platform data is not redistributed because of the platforms' terms; the scripts
download it again from the original sources.

## Run

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python run.py all                      # fetch, build, analyze, verify
    .venv/bin/python run.py fetch hourly             # one build step
    .venv/bin/python run.py analyze rq5              # one analysis
    .venv/bin/python run.py analyze final_tar final_rq3 final_band final_rules   # full estimators
    TP_END=2027-01-01 .venv/bin/python run.py all    # extend the sample

## Layout

    config.py        sample window, assets, sample months, event dates
    sources.py       downloaders (Indodax, Binance, Coinbase, Bank Indonesia JISDOR,
                     Yahoo USD/IDR, Tokocrypto legacy API, Upbit Indonesia)
    build.py         aligns the sources into panels in data/interim
    analysis/        rq1_rq2, rq3, rq3_pooled, rq4, rq5, rq6, xvenue, final (TAR, band, rules), extras
    verify.py        compares results/summary.json with expected.json

## Definitions

- Total premium: ln BTC/IDR − ln(BTC/USDT × USDT/USD × JISDOR), at the JISDOR fixing (10:00 WIB) on business days.
- Crypto channel: ln BTC/IDR − ln USDT/IDR − ln BTC/USDT. Rupiah channel: ln USDT/IDR − ln(USDT/USD × JISDOR).
- Hourly bars are labelled by their opening time (UTC); the close is the last trade before the bar ends.

## Notes

- A raw request window is cached only once it has closed, so reruns download only new data.
- Yahoo serves hourly USD/IDR for the last 730 days only. It is used only as a competing predictor.
- USDT/USD (Coinbase) starts in mid-2021; USDT is taken at parity before that.
- The decision rules for the January to April 2027 forward test are fixed in a pre-analysis plan
  registered on OSF before the test period.

## License

MIT, see LICENSE.
