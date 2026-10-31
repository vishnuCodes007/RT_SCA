"""Export the exact synthetic workload instances used in the reported
experiments (E1-E5) as CSV files in ../dataset/.

The generators, seeds and loop order are the same as in simulate.py, so the
exported slices are identical to the ones the allocators saw. Allocators are
not run, so the export takes a few seconds.
"""

import csv
import os
import random

import numpy as np

import simulate as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
UF = S.WORKLOADS["W2-balanced"]
BATCH_COLS = ["experiment", "workload", "load", "rho", "N", "trial", "hosts",
              "slice_id", "type", "demand", "sigma", "tau"]
DYN_COLS = ["experiment", "rho", "trial", "period", "slice_id", "type",
            "demand", "sigma", "tau", "hold"]


def batch_rows(exp, workload, load, rho, n, trial, batch):
    H = S.hosts_for_load(batch, rho)
    for s in batch:
        yield [exp, workload, load, rho, n, trial, H, s["id"], s["type"],
               round(s["d"], 6), round(s["sigma"], 6), round(s["tau"], 6)]


def write(name, cols, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = 0
        for r in rows:
            w.writerow(r)
            n += 1
    print(f"  {name}: {n} rows")


def e1():  # exp_scalability
    rng_w = random.Random(S.SEED)
    for n in [10, 25, 50, 100, 200, 400]:
        for t in range(12):
            yield from batch_rows("E1", "W2-balanced", "medium", 0.70, n, t,
                                  S.gen_batch(rng_w, n, UF))


def e2():  # exp_workloads
    rng_w = random.Random(S.SEED + 7)
    for w, uf in S.WORKLOADS.items():
        for ln, rho in {"low": 0.40, "medium": 0.70, "high": 0.95}.items():
            for t in range(12):
                yield from batch_rows("E2", w, ln, rho, 100, t, S.gen_batch(rng_w, 100, uf))


def e4():  # exp_ablation (also used by e4ref.py)
    rng_w = random.Random(S.SEED + 99)
    for t in range(20):
        yield from batch_rows("E4", "W2-balanced", "medium", 0.70, 100, t,
                              S.gen_batch(rng_w, 100, UF))


def e4_burst():  # deadline-guard burst inside exp_ablation
    rng_w2 = random.Random(S.SEED + 5)
    for t in range(10):
        yield from batch_rows("E4-burst", "W2-balanced", "medium", 0.70, 2000, t,
                              S.gen_batch(rng_w2, 2000, UF))


def e5():  # exp_stats
    rng_w = random.Random(S.SEED + 555)
    for t in range(30):
        yield from batch_rows("E5", "W2-balanced", "medium", 0.70, 100, t,
                              S.gen_batch(rng_w, 100, UF))


def e3(trials=5, periods=160, H=10, hold=8.0):
    """Arrival traces of dyn_run(). They depend only on (rho, trial), so the
    same trace is fed to every allocator and to the dynamic ablation."""
    E_d = UF * S.E_D_URLLC + (1 - UF) * S.E_D_EMBB
    for rho in [0.6, 0.8, 1.0, 1.2]:
        lam = rho * H * S.DOMAINS * S.CAPACITY / (E_d * hold)
        for tr in range(trials):
            rng_w = random.Random(S.SEED + 100 * tr)
            np_rng = np.random.default_rng(S.SEED + 100 * tr)
            sid = 0
            for t in range(periods):
                k = int(np_rng.poisson(lam))
                holds = np_rng.exponential(hold, size=k)
                for j in range(k):
                    s = S.gen_slice(rng_w, UF)
                    yield ["E3", rho, tr, t, sid, s["type"], round(s["d"], 6),
                           round(s["sigma"], 6), round(s["tau"], 6), round(float(holds[j]), 6)]
                    sid += 1


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Exporting dataset to", os.path.normpath(OUT))
    write("E1_scalability.csv", BATCH_COLS, e1())
    write("E2_workloads.csv", BATCH_COLS, e2())
    write("E4_ablation.csv", BATCH_COLS, e4())
    write("E4_burst_N2000.csv", BATCH_COLS, e4_burst())
    write("E5_statistics.csv", BATCH_COLS, e5())
    write("E3_dynamic_arrivals.csv", DYN_COLS, e3())
