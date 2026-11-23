"""Compare results/summary.json with the pilot values reported in the draft (expected.json)."""
import json

import pandas as pd

from config import RESULTS, ROOT


def run():
    exp = json.loads((ROOT / "expected.json").read_text())
    got = json.loads((RESULTS / "summary.json").read_text())
    rows = []
    for k, v in exp.items():
        if k.startswith("_"):
            continue
        target, tol = v
        x = got.get(k)
        ok = x is not None and abs(float(x) - target) <= tol
        rows.append(dict(key=k, pilot=target, pipeline=x, tolerance=tol, ok=ok))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "verify.csv", index=False)
    bad = T[~T.ok]
    print(f"{T.ok.sum()} of {len(T)} pilot numbers reproduced within tolerance")
    if len(bad):
        print(bad.to_string(index=False))
    return T


if __name__ == "__main__":
    run()
