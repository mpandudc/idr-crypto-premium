"""Command-line entry point.

  python run.py fetch [jisdor hourly alts m1 xvenue]   download (cached) and build panels
  python run.py analyze [rq1 rq3 rq4 rq5 rq6 xvenue]    run analyses (default: all)
  python run.py analyze final_tar final_rules           full estimators (slow)
  python run.py analyze final_rq3                       pooled VECM per month and session
  python run.py verify                                  compare with the pilot values
  python run.py all                                     everything, in order
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import build  # noqa: E402
from analysis import common  # noqa: E402
from config import RESULTS  # noqa: E402

ALL_ANALYSES = ["rq1", "rq3", "rq4", "rq5", "rq6", "xvenue"]


def analyze(which):
    RESULTS.mkdir(parents=True, exist_ok=True)
    H, D, J = build.load_panels()
    print("panels: hourly", len(H), "| daily", len(D), D.index.min().date(), "to", D.index.max().date(), flush=True)
    for name in which:
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        if name == "rq1":
            from analysis import rq1_rq2
            rq1_rq2.run(H, D, J)
        elif name == "rq3":
            from analysis import rq3
            rq3.run()
        elif name == "rq4":
            from analysis import rq4
            rq4.run(H, D, J)
        elif name == "rq5":
            from analysis import rq5
            rq5.run(H, D, J, build.load_alts())
        elif name == "rq6":
            from analysis import rq6
            rq6.run(H, D, J)
        elif name == "final_tar":
            from analysis import final
            final.run_tar(H, D)
        elif name == "final_rq3":
            from analysis import rq3_pooled
            rq3_pooled.run()
        elif name == "final_rules":
            from analysis import final
            final.run_rules(H, D)
        elif name == "xvenue":
            from analysis import xvenue
            xvenue.run(H, D, J, build.load_xvenue())
        common.save_summary()
        print(f"({name} {time.time() - t0:.0f}s)", flush=True)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    args = argv[2:]
    if cmd == "fetch":
        build.run(args or ("jisdor", "hourly", "alts", "m1", "xvenue"))
    elif cmd == "analyze":
        analyze(args or ALL_ANALYSES)
    elif cmd == "verify":
        import verify
        verify.run()
    elif cmd == "all":
        build.run()
        analyze(ALL_ANALYSES)
        import verify
        verify.run()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
