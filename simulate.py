"""
RT-SCA simulation harness (v3).

Discrete-event model of a near-RT RIC managing slice placement over an
O-Cloud resource pool. Each host has D shared cache/NUMA domains; slices
co-resident in the same domain (strong leakage, LAMBDA_C) or on the same host
(weaker cross-domain leakage, LAMBDA_H) contribute side-channel risk

    r(i, j) = lambda * (sigma_i * tau_j + sigma_j * tau_i)

where sigma = victim sensitivity and tau = probability the tenant is
adversarial.

Allocators
  FF      : security-unaware first-fit (consolidating baseline)
  ISO     : hard trust-zone isolation (URLLC-only domains, no spill)
  SCA-OPT : centralised offline optimiser in the style of SCA-RA:
            exhaustive slot search + local search + restarts,
            lexicographic objective (max admitted, then min risk), no time
            budget. An approximation of the SCA-RA style, NOT Li et al.'s code.
  RT-SCA  : proposed. A1-provided host reservation ratio (soft trust zones),
            O(1) marginal risk from sufficient statistics, bounded
            K-candidate search with best-fit tie-break, controlled spill-over,
            deadline guard that defers the remainder to the next loop.

All results are SIMULATED in single-threaded Python on a commodity CPU.
"""

import json
import math
import random
import time

import numpy as np
from scipy import stats

SEED = 2026
LAMBDA_C = 1.0
LAMBDA_H = 0.2
CAPACITY = 100.0
DOMAINS = 4
SENS_THETA = 0.5          # sigma >= theta -> sensitive class
UNTRUSTED_TAU = 0.3
MARGIN = 0.25             # reservation safety margin
E_D_URLLC, E_D_EMBB = 5.0, 12.5

WORKLOADS = {"W1-URLLC-heavy": 0.70, "W2-balanced": 0.40, "W3-eMBB-heavy": 0.15}


def gen_slice(rng, uf):
    if rng.random() < uf:
        return {"type": "URLLC", "d": rng.uniform(2, 8),
                "sigma": rng.uniform(0.7, 1.0), "tau": rng.uniform(0.0, 0.1)}
    return {"type": "eMBB", "d": rng.uniform(5, 20),
            "sigma": rng.uniform(0.1, 0.4), "tau": rng.uniform(0.1, 0.6)}


def gen_batch(rng, n, uf):
    return [dict(gen_slice(rng, uf), id=i) for i in range(n)]


def hosts_for_load(batch, rho):
    return max(2, math.ceil(sum(s["d"] for s in batch) / (rho * DOMAINS * CAPACITY)))


def forecast_ratio(uf, error=0.0):
    """Reservation ratio the non-RT RIC would push over A1, from the workload
    profile's expected sensitive-demand share (optionally mis-forecast)."""
    share = uf * E_D_URLLC / (uf * E_D_URLLC + (1 - uf) * E_D_EMBB)
    return min(1.0, max(0.0, share * (1 + MARGIN) * (1 + error)))


# ----------------------------------------------------------------------
# Pool with sufficient statistics (pure-Python lists for O(1) access)
# ----------------------------------------------------------------------

class Pool:
    def __init__(self, H):
        self.H = H
        n = H * DOMAINS
        self.load = [0.0] * n
        self.S = [0.0] * n
        self.U = [0.0] * n
        self.Sh = [0.0] * H
        self.Uh = [0.0] * H
        self.nsens = [0] * n
        self.res = [[] for _ in range(n)]

    def copy(self):
        p = Pool.__new__(Pool)
        p.H = self.H
        p.load, p.S, p.U = self.load[:], self.S[:], self.U[:]
        p.Sh, p.Uh, p.nsens = self.Sh[:], self.Uh[:], self.nsens[:]
        p.res = [r[:] for r in self.res]
        return p

    def adopt(self, o):
        self.load, self.S, self.U = o.load, o.S, o.U
        self.Sh, self.Uh, self.nsens, self.res = o.Sh, o.Uh, o.nsens, o.res

    def feasible(self, s, i):
        return self.load[i] + s["d"] <= CAPACITY + 1e-9

    def add(self, s, i):
        h = i // DOMAINS
        self.load[i] += s["d"]; self.S[i] += s["sigma"]; self.U[i] += s["tau"]
        self.Sh[h] += s["sigma"]; self.Uh[h] += s["tau"]
        if s["sigma"] >= SENS_THETA:
            self.nsens[i] += 1
        self.res[i].append(s)

    def remove(self, s, i):
        h = i // DOMAINS
        self.load[i] -= s["d"]; self.S[i] -= s["sigma"]; self.U[i] -= s["tau"]
        self.Sh[h] -= s["sigma"]; self.Uh[h] -= s["tau"]
        if s["sigma"] >= SENS_THETA:
            self.nsens[i] -= 1
        for k, x in enumerate(self.res[i]):
            if x is s:
                del self.res[i][k]
                break

    def marginal_agg(self, s, i):                       # O(1)
        h = i // DOMAINS
        Sd, Ud = self.S[i], self.U[i]
        return (LAMBDA_C * (s["sigma"] * Ud + s["tau"] * Sd)
                + LAMBDA_H * (s["sigma"] * (self.Uh[h] - Ud) + s["tau"] * (self.Sh[h] - Sd)))

    def marginal_pairwise(self, s, i):                  # O(m_h)
        h = i // DOMAINS
        r = 0.0
        for j in range(h * DOMAINS, (h + 1) * DOMAINS):
            lam = LAMBDA_C if j == i else LAMBDA_H
            for x in self.res[j]:
                r += lam * (s["sigma"] * x["tau"] + x["sigma"] * s["tau"])
        return r

    def total_risk(self):
        R = 0.0
        for h in range(self.H):
            within, host_self = 0.0, 0.0
            for i in range(h * DOMAINS, (h + 1) * DOMAINS):
                st = sum(x["sigma"] * x["tau"] for x in self.res[i])
                within += self.S[i] * self.U[i] - st
                host_self += st
            host_all = self.Sh[h] * self.Uh[h] - host_self
            R += LAMBDA_C * within + LAMBDA_H * (host_all - within)
        return R

    def uer(self):
        n_u = n_e = 0
        for members in self.res:
            untrusted = any(x["tau"] > UNTRUSTED_TAU for x in members)
            for x in members:
                if x["type"] == "URLLC":
                    n_u += 1
                    n_e += untrusted
        return n_e / n_u if n_u else 0.0


# ----------------------------------------------------------------------
# Allocators -> dict(adm, blk, dfr)
# ----------------------------------------------------------------------

def alloc_ff(pool, batch, rng=None, **_):
    adm = blk = 0
    n = pool.H * DOMAINS
    for s in batch:
        for i in range(n):
            if pool.feasible(s, i):
                pool.add(s, i); s["loc"] = i; adm += 1
                break
        else:
            s["loc"] = None; blk += 1
    return {"adm": adm, "blk": blk, "dfr": 0}


def alloc_iso(pool, batch, rng=None, **_):
    adm = blk = 0
    n = pool.H * DOMAINS
    for s in batch:
        sens = s["sigma"] >= SENS_THETA
        for i in range(n):
            empty = not pool.res[i]
            if sens:
                ok = empty or pool.nsens[i] == len(pool.res[i])
            else:
                ok = pool.nsens[i] == 0
            if ok and pool.feasible(s, i):
                pool.add(s, i); s["loc"] = i; adm += 1
                break
        else:
            s["loc"] = None; blk += 1
    return {"adm": adm, "blk": blk, "dfr": 0}


def _best_pairwise(p, s, n):
    best, bk = None, None
    for i in range(n):
        if p.feasible(s, i):
            k = (p.marginal_pairwise(s, i), -p.load[i])
            if bk is None or k < bk:
                best, bk = i, k
    return best, (bk[0] if bk else None)


def alloc_sca_opt(pool, batch, rng, restarts=3, max_passes=20, **_):
    base = pool.copy()
    n = pool.H * DOMAINS
    best = None
    for r in range(restarts):
        p = base.copy()
        jit = 0.0 if r == 0 else 0.05
        order = sorted(batch, key=lambda s: -(s["sigma"] + rng.uniform(0, jit)))
        loc = {}
        for s in order:
            i, _ = _best_pairwise(p, s, n)
            loc[s["id"]] = i
            if i is not None:
                p.add(s, i)
        for _ in range(max_passes):
            improved = False
            for s in order:
                cur = loc[s["id"]]
                if cur is None:
                    i, _ = _best_pairwise(p, s, n)
                    if i is not None:
                        p.add(s, i); loc[s["id"]] = i; improved = True
                    continue
                p.remove(s, cur)
                cur_cost = p.marginal_pairwise(s, cur)
                i, c = _best_pairwise(p, s, n)
                if i is not None and c < cur_cost - 1e-9:
                    p.add(s, i); loc[s["id"]] = i; improved = True
                else:
                    p.add(s, cur)
            if not improved:
                break
        adm = sum(v is not None for v in loc.values())
        obj = (-adm, p.total_risk())
        if best is None or obj < best[0]:
            best = (obj, p, loc)
    _, p, loc = best
    pool.adopt(p)
    adm = blk = 0
    for s in batch:
        s["loc"] = loc[s["id"]]
        adm += s["loc"] is not None
        blk += s["loc"] is None
    return {"adm": adm, "blk": blk, "dfr": 0}


def alloc_rt_sca(pool, batch, rng, K=8, ratio=0.26, budget_ms=None,
                 use_zone=True, use_risk=True, use_agg=True, use_order=True,
                 eps=0.05, gran="domain", **_):
    t0 = time.perf_counter()
    H = pool.H
    n = H * DOMAINS
    if not use_zone:
        Zr = 0
    elif gran == "host":
        Zr = min(H, max(1, math.ceil(ratio * H))) * DOMAINS
    else:                                # domain-level granularity
        Zr = min(n, max(1, round(ratio * n)))
    zone_sens = (0, Zr)                  # reserved slots (contiguous)
    zone_open = (Zr, n)
    if Zr == 0 or Zr == n:               # degenerate: single zone
        zone_sens = zone_open = (0, n)
    f = pool.marginal_agg if use_agg else pool.marginal_pairwise
    order = sorted(batch, key=lambda s: -s["sigma"]) if use_order else list(batch)
    adm = blk = dfr = 0
    deferred = []

    def ok_for(s, i, sens):
        if not pool.feasible(s, i):
            return False
        if not sens and use_zone and zone_sens != zone_open and i < zone_sens[1]:
            return pool.nsens[i] == 0     # untrusted spill: never join sensitive
        return True

    def pick(s, zone, sens):
        lo, hi = zone
        size = hi - lo
        if size <= 0:
            return None
        k = min(K, size)
        cand = [lo + c for c in rng.sample(range(size), k)]
        feas = [i for i in cand if ok_for(s, i, sens)]
        if not feas:
            start = rng.randrange(size)          # rotating linear probe
            for off in range(size):
                i = lo + (start + off) % size
                if ok_for(s, i, sens):
                    feas = [i]
                    break
        if not feas:
            return None
        if use_risk:
            # epsilon-tolerant best-fit: among candidates whose marginal
            # risk is within eps of the minimum, pick the fullest slot
            scored = [(f(s, i), i) for i in feas]
            rmin = min(r for r, _ in scored)
            return max((i for r, i in scored if r <= rmin + eps),
                       key=lambda i: pool.load[i])
        return rng.choice(feas)

    for idx, s in enumerate(order):
        if budget_ms is not None and (time.perf_counter() - t0) * 1000.0 > 0.9 * budget_ms:
            deferred = order[idx:]
            break
        sens = s["sigma"] >= SENS_THETA
        own, other = (zone_sens, zone_open) if sens else (zone_open, zone_sens)
        i = pick(s, own, sens)
        if i is None and other != own:
            i = pick(s, other, sens)              # controlled spill-over
        if i is None:
            s["loc"] = None; blk += 1
        else:
            pool.add(s, i); s["loc"] = i; adm += 1
    for s in deferred:
        s["loc"] = "deferred"
    dfr = len(deferred)
    return {"adm": adm, "blk": blk, "dfr": dfr, "deferred": deferred}


METHODS = ["FF", "ISO", "SCA-OPT", "RT-SCA"]
ALLOC = {"FF": alloc_ff, "ISO": alloc_iso, "SCA-OPT": alloc_sca_opt, "RT-SCA": alloc_rt_sca}


def run(m, pool, batch, rng, **kw):
    t0 = time.perf_counter()
    r = ALLOC[m](pool, batch, rng, **kw)
    r["lat"] = (time.perf_counter() - t0) * 1000.0
    return r


def clone(batch):
    return [dict(s) for s in batch]


def mstd(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


# ----------------------------------------------------------------------
# E1 scalability
# ----------------------------------------------------------------------

def exp_scalability(trials=12):
    sizes = [10, 25, 50, 100, 200, 400]
    rng_w = random.Random(SEED)
    uf = WORKLOADS["W2-balanced"]
    out = {"N": sizes, "data": {m: {k: [] for k in
           ["lat", "lat_sd", "p95", "re", "re_sd", "adm", "uer", "miss10"]} for m in METHODS}}
    for n in sizes:
        acc = {m: {"lat": [], "re": [], "adm": [], "uer": []} for m in METHODS}
        for t in range(trials):
            batch = gen_batch(rng_w, n, uf)
            H = hosts_for_load(batch, 0.70)
            ref = Pool(H); alloc_ff(ref, clone(batch)); Rff = max(ref.total_risk(), 1e-9)
            for m in METHODS:
                pool = Pool(H); rng = random.Random(SEED + 1000 * t + n)
                kw = {"K": 8, "ratio": forecast_ratio(uf)} if m == "RT-SCA" else {}
                r = run(m, pool, clone(batch), rng, **kw)
                acc[m]["lat"].append(r["lat"]); acc[m]["re"].append(pool.total_risk() / Rff)
                acc[m]["adm"].append(r["adm"] / n); acc[m]["uer"].append(pool.uer())
        for m in METHODS:
            L = np.array(acc[m]["lat"]); D = out["data"][m]
            D["lat"].append(float(L.mean())); D["lat_sd"].append(float(L.std(ddof=1)))
            D["p95"].append(float(np.percentile(L, 95)))
            a, b = mstd(acc[m]["re"]); D["re"].append(a); D["re_sd"].append(b)
            D["adm"].append(float(np.mean(acc[m]["adm"]))); D["uer"].append(float(np.mean(acc[m]["uer"])))
            D["miss10"].append(float(np.mean(L > 10.0)))
        print(f"  E1 N={n}", flush=True)
    return out


# ----------------------------------------------------------------------
# E2 workloads x load
# ----------------------------------------------------------------------

def exp_workloads(trials=12, n=100):
    loads = {"low": 0.40, "medium": 0.70, "high": 0.95}
    rng_w = random.Random(SEED + 7)
    out = {}
    for w, uf in WORKLOADS.items():
        for ln, rho in loads.items():
            acc = {m: {"lat": [], "re": [], "adm": [], "uer": []} for m in METHODS}
            for t in range(trials):
                batch = gen_batch(rng_w, n, uf)
                H = hosts_for_load(batch, rho)
                ref = Pool(H); alloc_ff(ref, clone(batch)); Rff = max(ref.total_risk(), 1e-9)
                for m in METHODS:
                    pool = Pool(H); rng = random.Random(SEED + 31 * t)
                    kw = {"K": 8, "ratio": forecast_ratio(uf)} if m == "RT-SCA" else {}
                    r = run(m, pool, clone(batch), rng, **kw)
                    acc[m]["lat"].append(r["lat"]); acc[m]["re"].append(pool.total_risk() / Rff)
                    acc[m]["adm"].append(r["adm"] / n); acc[m]["uer"].append(pool.uer())
            out[f"{w}|{ln}"] = {m: {k: float(np.mean(v)) for k, v in acc[m].items()} for m in METHODS}
        print(f"  E2 {w}", flush=True)
    return out


# ----------------------------------------------------------------------
# E3 dynamic arrivals
# ----------------------------------------------------------------------

def dyn_run(m, rho, kw, trials=5, periods=160, warmup=40, H=10, seed=SEED):
    uf = WORKLOADS["W2-balanced"]
    E_d = uf * E_D_URLLC + (1 - uf) * E_D_EMBB
    hold = 8.0
    lam = rho * H * DOMAINS * CAPACITY / (E_d * hold)
    bps, ubps, rps, ues, lats = [], [], [], [], []
    for tr in range(trials):
        rng_w = random.Random(seed + 100 * tr)
        np_rng = np.random.default_rng(seed + 100 * tr)
        rng = random.Random(seed + 7 * tr)
        pool = Pool(H)
        active, carry = [], []
        req = blocked = ureq = ublk = 0
        sid = 0
        rs, us = [], []
        for t in range(periods):
            keep = []
            for rel, s in active:
                if rel <= t:
                    pool.remove(s, s["loc"])
                else:
                    keep.append((rel, s))
            active = keep
            k = int(np_rng.poisson(lam))
            holds = np_rng.exponential(hold, size=k)
            new = []
            for j in range(k):
                s = gen_slice(rng_w, uf); s["id"] = sid; s["hold"] = holds[j]; sid += 1
                new.append(s)
            batch = carry + new
            carry = []
            if not batch:
                continue
            r = run(m, pool, batch, rng, **kw)
            for s in batch:
                if isinstance(s.get("loc"), int):
                    active.append((t + max(1, int(round(s["hold"]))), s))
            carry = r.get("deferred", [])
            if t >= warmup:
                req += len(new); blocked += r["blk"]
                ureq += sum(s["type"] == "URLLC" for s in new)
                ublk += sum(s["type"] == "URLLC" and s.get("loc") is None for s in new)
                lats.append(r["lat"])
                if active:
                    rs.append(pool.total_risk() / len(active)); us.append(pool.uer())
        bps.append(blocked / max(1, req)); ubps.append(ublk / max(1, ureq))
        rps.append(np.mean(rs)); ues.append(np.mean(us))
    return {"block": bps, "ublock": ubps, "risk": rps, "uer": ues, "lat": lats}


def method_kw(m, budget=10.0):
    uf = WORKLOADS["W2-balanced"]
    if m == "RT-SCA":
        return {"K": 8, "ratio": forecast_ratio(uf), "budget_ms": budget}
    if m == "SCA-OPT":
        return {"restarts": 2, "max_passes": 5}
    return {}


def exp_dynamic():
    offered = [0.6, 0.8, 1.0, 1.2]
    keys = ["block", "block_sd", "ublock", "risk", "uer", "lat", "miss10"]
    out = {"offered": offered, "data": {m: {k: [] for k in keys} for m in METHODS}}
    raw = {}
    for rho in offered:
        for m in METHODS:
            r = dyn_run(m, rho, method_kw(m))
            raw[f"{m}|{rho}"] = r["block"]
            D = out["data"][m]
            a, b = mstd(r["block"]); D["block"].append(a); D["block_sd"].append(b)
            D["ublock"].append(float(np.mean(r["ublock"])))
            D["risk"].append(float(np.mean(r["risk"]))); D["uer"].append(float(np.mean(r["uer"])))
            D["lat"].append(float(np.mean(r["lat"])))
            D["miss10"].append(float(np.mean(np.array(r["lat"]) > 10.0)))
        print(f"  E3 rho={rho}", flush=True)
    return out, raw


def exp_ablation_dynamic():
    variants = {
        "RT-SCA (full)": {},
        "w/o sigma-ordering": {"use_order": False},
        "host-level zoning": {"gran": "host"},
        "w/o trust zoning": {"use_zone": False},
    }
    out = {}
    for v, extra in variants.items():
        out[v] = {}
        for rho in [1.0, 1.2]:
            kw = method_kw("RT-SCA"); kw.update(extra)
            r = dyn_run("RT-SCA", rho, kw)
            out[v][str(rho)] = {"block": float(np.mean(r["block"])),
                                "ublock": float(np.mean(r["ublock"])),
                                "risk": float(np.mean(r["risk"])),
                                "uer": float(np.mean(r["uer"]))}
    return out


# ----------------------------------------------------------------------
# E4 ablation
# ----------------------------------------------------------------------

def exp_ablation(trials=20, n=100):
    uf = WORKLOADS["W2-balanced"]
    rho_r = forecast_ratio(uf)
    rng_w = random.Random(SEED + 99)
    variants = {
        "RT-SCA (full)": {},
        "w/o trust zoning": {"use_zone": False},
        "w/o risk scoring": {"use_risk": False},
        "w/o O(1) aggregates": {"use_agg": False},
        "w/o sigma-ordering": {"use_order": False},
        "host-level zoning": {"gran": "host"},
        "eps = 0 (pure min-risk)": {"eps": 0.0},
        "forecast -50%": {"ratio": forecast_ratio(uf, -0.5)},
        "forecast +50%": {"ratio": forecast_ratio(uf, +0.5)},
    }
    Ks = [1, 2, 4, 8, 16, 32, 64]
    acc = {v: {"lat": [], "re": [], "uer": [], "adm": []} for v in variants}
    kacc = {K: {"lat": [], "re": [], "uer": []} for K in Ks}
    batches = []
    for t in range(trials):
        batch = gen_batch(rng_w, n, uf)
        H = hosts_for_load(batch, 0.70)
        batches.append((batch, H))
        ref = Pool(H); alloc_ff(ref, clone(batch)); Rff = max(ref.total_risk(), 1e-9)
        for v, kw in variants.items():
            args = {"K": 8, "ratio": rho_r}; args.update(kw)
            pool = Pool(H); rng = random.Random(SEED + t)
            r = run("RT-SCA", pool, clone(batch), rng, **args)
            acc[v]["lat"].append(r["lat"]); acc[v]["re"].append(pool.total_risk() / Rff)
            acc[v]["uer"].append(pool.uer()); acc[v]["adm"].append(r["adm"] / n)
        for K in Ks:
            pool = Pool(H); rng = random.Random(SEED + t)
            r = run("RT-SCA", pool, clone(batch), rng, K=K, ratio=rho_r)
            kacc[K]["lat"].append(r["lat"]); kacc[K]["re"].append(pool.total_risk() / Rff)
            kacc[K]["uer"].append(pool.uer())
    ab = {}
    for v in variants:
        ab[v] = {k: float(np.mean(x)) for k, x in acc[v].items()}
        ab[v]["re_sd"] = float(np.std(acc[v]["re"], ddof=1))
    ks = {"K": Ks, "lat": [float(np.mean(kacc[K]["lat"])) for K in Ks],
          "re": [float(np.mean(kacc[K]["re"])) for K in Ks],
          "uer": [float(np.mean(kacc[K]["uer"])) for K in Ks]}

    # Deadline guard on a large burst (N = 2000)
    guard = {}
    for label, budget in [("with guard (B = 10 ms)", 10.0), ("without guard", None)]:
        rng_w2 = random.Random(SEED + 5)
        lats, res, dfr = [], [], []
        for t in range(10):
            batch = gen_batch(rng_w2, 2000, uf)
            H = hosts_for_load(batch, 0.70)
            ref = Pool(H); alloc_ff(ref, clone(batch)); Rff = max(ref.total_risk(), 1e-9)
            pool = Pool(H); rng = random.Random(SEED + t)
            r = run("RT-SCA", pool, clone(batch), rng, K=8, ratio=rho_r, budget_ms=budget)
            lats.append(r["lat"]); res.append(pool.total_risk() / Rff); dfr.append(r["dfr"] / 2000)
        guard[label] = {"lat_mean": float(np.mean(lats)), "lat_max": float(np.max(lats)),
                        "re_admitted": float(np.mean(res)), "deferred_frac": float(np.mean(dfr))}
    return ab, ks, guard


# ----------------------------------------------------------------------
# E5 statistics
# ----------------------------------------------------------------------

def d_paired(a, b):
    x = np.array(a) - np.array(b)
    return float(x.mean() / x.std(ddof=1))


def d_ind(a, b):
    a, b = np.array(a), np.array(b)
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def ci95(x):
    x = np.array(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def exp_stats(dyn_raw, trials=30, n=100):
    uf = WORKLOADS["W2-balanced"]
    rng_w = random.Random(SEED + 555)
    lat = {m: [] for m in METHODS}; re = {m: [] for m in METHODS}; ue = {m: [] for m in METHODS}
    for t in range(trials):
        batch = gen_batch(rng_w, n, uf)
        H = hosts_for_load(batch, 0.70)
        ref = Pool(H); alloc_ff(ref, clone(batch)); Rff = max(ref.total_risk(), 1e-9)
        for m in METHODS:
            pool = Pool(H); rng = random.Random(SEED + t)
            kw = {"K": 8, "ratio": forecast_ratio(uf)} if m == "RT-SCA" else {}
            r = run(m, pool, clone(batch), rng, **kw)
            lat[m].append(r["lat"]); re[m].append(pool.total_risk() / Rff); ue[m].append(pool.uer())
    S = {"n_trials": trials}
    t_, p_ = stats.ttest_rel(lat["SCA-OPT"], lat["RT-SCA"])
    w_, pw = stats.wilcoxon(lat["SCA-OPT"], lat["RT-SCA"])
    S["lat_opt_vs_rt"] = {"t": float(t_), "p": float(p_), "W": float(w_), "pw": float(pw),
                          "d": d_paired(lat["SCA-OPT"], lat["RT-SCA"]),
                          "ci_opt": ci95(lat["SCA-OPT"]), "ci_rt": ci95(lat["RT-SCA"])}
    t_, p_ = stats.ttest_ind(re["FF"], re["RT-SCA"], equal_var=False)
    S["re_ff_vs_rt"] = {"t": float(t_), "p": float(p_), "d": d_ind(re["FF"], re["RT-SCA"]),
                        "ci_rt": ci95(re["RT-SCA"])}
    t_, p_ = stats.ttest_rel(re["ISO"], re["RT-SCA"])
    w_, pw = stats.wilcoxon(re["ISO"], re["RT-SCA"])
    S["re_iso_vs_rt"] = {"t": float(t_), "p": float(p_), "W": float(w_), "pw": float(pw),
                         "d": d_paired(re["ISO"], re["RT-SCA"]), "ci_iso": ci95(re["ISO"])}
    t_, p_ = stats.ttest_rel(re["RT-SCA"], re["SCA-OPT"])
    w_, pw = stats.wilcoxon(re["RT-SCA"], re["SCA-OPT"])
    S["re_rt_vs_opt"] = {"t": float(t_), "p": float(p_), "W": float(w_), "pw": float(pw),
                         "gap": float(np.mean(re["RT-SCA"]) - np.mean(re["SCA-OPT"])),
                         "ci_opt": ci95(re["SCA-OPT"])}
    S["uer_means"] = {m: float(np.mean(ue[m])) for m in METHODS}
    groups = [dyn_raw[f"{m}|1.0"] for m in METHODS]
    F, pF = stats.f_oneway(*groups)
    Hk, pk = stats.kruskal(*groups)
    S["block_anova_rho1"] = {"F": float(F), "p": float(pF), "H": float(Hk), "pk": float(pk)}
    t_, p_ = stats.ttest_ind(dyn_raw["ISO|1.0"], dyn_raw["RT-SCA|1.0"], equal_var=False)
    S["block_iso_vs_rt_rho1"] = {"t": float(t_), "p": float(p_)}
    t_, p_ = stats.ttest_ind(dyn_raw["ISO|1.2"], dyn_raw["RT-SCA|1.2"], equal_var=False)
    S["block_iso_vs_rt_rho1.2"] = {"t": float(t_), "p": float(p_)}
    return S


if __name__ == "__main__":
    import sys
    random.seed(SEED); np.random.seed(SEED)
    T0 = time.time()
    res = {}
    print("E1", flush=True); res["E1"] = exp_scalability()
    print("E2", flush=True); res["E2"] = exp_workloads()
    print("E4", flush=True); res["E4"], res["E4K"], res["E4guard"] = exp_ablation()
    print("E3", flush=True); res["E3"], res["E3raw"] = exp_dynamic()
    print("E4dyn", flush=True); res["E4dyn"] = exp_ablation_dynamic()
    print("E5", flush=True); res["E5"] = exp_stats(res["E3raw"])
    res["runtime_s"] = time.time() - T0
    with open("results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"done {res['runtime_s']:.0f}s", flush=True)
