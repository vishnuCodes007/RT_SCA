# RT-SCA: Real-Time Side-Channel-Aware Resource Allocation

Code and dataset for the paper **"RT-SCA: A Real-Time Side-Channel-Aware Resource
Allocation xApp for Near-RT RIC in O-RAN Networks"** by Vishnu Bhrigu, Department of
Computer Science.

The repository contains a discrete-event simulator of a near-RT RIC that places
network slices on an O-Cloud resource pool. It compares four allocators:

| Allocator | Description |
|---|---|
| FF | Security-unaware first-fit (baseline) |
| ISO | Hard trust-zone isolation |
| SCA-OPT | Offline optimiser in the style of SCA-RA (an approximation, not the original authors' code) |
| RT-SCA | Proposed real-time allocator with soft trust zones, O(1) risk scoring, bounded search and a deadline guard |

All results are simulated in single-threaded Python.

## Repository layout

```
simulation/
  simulate.py        simulator, allocators and all experiments (writes results.json)
  e4ref.py           baselines on the ablation instances (adds E4ref to results.json)
  calibrate.py       hyper-parameter calibration on a separate seed (777001)
  check.py           correctness checks (O(1) marginal risk vs brute force)
  export_dataset.py  writes the exact workload instances to ../dataset/
  results.json       raw results reported in the paper
dataset/             synthetic workload instances used in the paper (CSV)
```

## Running

Requires Python 3.9 or later.

```
pip install -r requirements.txt
cd simulation
python check.py            # correctness checks
python simulate.py         # all experiments, about 3 minutes on a 2-core CPU
python e4ref.py            # ablation baselines
python export_dataset.py   # regenerate the CSV files in ../dataset/
```

Seeds are fixed: 2026 for the reported experiments and 777001 for calibration.
Rerunning `simulate.py` reproduces every non-timing value in `results.json`
exactly. Decision latencies, and the deadline-guard results that depend on them,
vary with the CPU.

## Dataset

No public dataset labels slice-level side-channel exposure, so all workloads are
synthetic. The `dataset/` folder holds the exact slice instances that the
allocators saw in each experiment.

| File | Experiment | Rows |
|---|---|---|
| `E1_scalability.csv` | Decision time vs batch size (N = 10 to 400, 12 batches each) | 9,420 |
| `E2_workloads.csv` | 3 workloads x 3 loads, 12 batches of 100 slices each | 10,800 |
| `E3_dynamic_arrivals.csv` | Poisson arrival traces, offered load 0.6 to 1.2, 5 runs x 160 periods | 151,077 |
| `E4_ablation.csv` | Ablation and K sweep, 20 batches of 100 slices | 2,000 |
| `E4_burst_N2000.csv` | Deadline-guard burst, 10 batches of 2,000 slices | 20,000 |
| `E5_statistics.csv` | Statistical tests, 30 batches of 100 slices | 3,000 |

Batch files have these columns: `experiment, workload, load, rho, N, trial,
hosts, slice_id, type, demand, sigma, tau`.

`E3_dynamic_arrivals.csv` has these columns: `experiment, rho, trial, period,
slice_id, type, demand, sigma, tau, hold`.

| Column | Meaning |
|---|---|
| `type` | URLLC or eMBB |
| `demand` | Compute demand in capacity units (one domain holds 100) |
| `sigma` | Victim sensitivity |
| `tau` | Probability that the tenant is adversarial |
| `hosts` | Number of hosts sized for the target utilisation `rho` |
| `hold` | Holding time in control periods |

The same arrival trace for a given (rho, trial) is fed to every allocator.

## Note on AI assistance

Generative AI tools assisted in drafting the simulation code. The author reviewed
the code and takes responsibility for it, as stated in the paper's declarations.
