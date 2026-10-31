"""Hyper-parameter calibration on a SEPARATE seed set (CAL_SEED), so that
the final reported experiments (SEED) are not tuned on their own data."""
import random
import numpy as np
import simulate as S

CAL_SEED = 777001
uf = S.WORKLOADS["W2-balanced"]


def batch_eval(kw, trials=10, n=100, rho=0.7):
    rngw = random.Random(CAL_SEED)
    re, ue, adm = [], [], []
    for t in range(trials):
        b = S.gen_batch(rngw, n, uf); H = S.hosts_for_load(b, rho)
        ref = S.Pool(H); S.alloc_ff(ref, S.clone(b)); Rff = ref.total_risk()
        p = S.Pool(H)
        r = S.run("RT-SCA", p, S.clone(b), random.Random(CAL_SEED + t), K=8,
                  ratio=S.forecast_ratio(uf), **kw)
        re.append(p.total_risk() / Rff); ue.append(p.uer()); adm.append(r["adm"] / n)
    return np.mean(re), np.mean(ue), np.mean(adm)


def dyn_eval(kw, rho, trials=3, periods=160, warmup=40, H=10):
    E_d = uf * S.E_D_URLLC + (1 - uf) * S.E_D_EMBB
    lam = rho * H * S.DOMAINS * S.CAPACITY / (E_d * 8.0)
    bl, ubl, rs = [], [], []
    for tr in range(trials):
        rngw = random.Random(CAL_SEED + 100 * tr); nr = np.random.default_rng(CAL_SEED + tr)
        rng = random.Random(CAL_SEED + 7 * tr); pool = S.Pool(H)
        active, carry = [], []; req = blk = ureq = ublk = 0; sid = 0; rr = []
        for t in range(periods):
            keep = []
            for rel, s in active:
                if rel <= t: pool.remove(s, s["loc"])
                else: keep.append((rel, s))
            active = keep
            k = int(nr.poisson(lam)); hs = nr.exponential(8.0, size=k)
            new = []
            for j in range(k):
                s = S.gen_slice(rngw, uf); s["id"] = sid; s["hold"] = hs[j]; sid += 1; new.append(s)
            batch = carry + new
            if not batch: continue
            r = S.run("RT-SCA", pool, batch, rng, K=8, ratio=S.forecast_ratio(uf), budget_ms=10.0, **kw)
            for s in batch:
                if isinstance(s.get("loc"), int):
                    active.append((t + max(1, int(round(s["hold"]))), s))
            carry = r.get("deferred", [])
            if t >= warmup:
                req += len(new); blk += r["blk"]
                ureq += sum(s["type"] == "URLLC" for s in new)
                ublk += sum(s["type"] == "URLLC" and s.get("loc") is None for s in new)
                if active: rr.append(pool.total_risk() / len(active))
        bl.append(blk / req); ubl.append(ublk / max(1, ureq)); rs.append(np.mean(rr))
    return np.mean(bl), np.mean(ubl), np.mean(rs)


configs = []
for gran in ["host", "domain"]:
    for eps in [0.0, 0.05, 0.1, 0.2, 0.3]:
        configs.append(dict(gran=gran, eps=eps))
configs.append(dict(gran="domain", eps=0.1, use_order=False))

print(f"{'config':40s} {'RE':>6s} {'UER':>6s} {'adm':>6s} | " +
      " | ".join(f"rho={r}: blk  uBlk  risk" for r in [0.8, 1.0, 1.2]))
for c in configs:
    re, ue, adm = batch_eval(c)
    dyn = [dyn_eval(c, r) for r in [0.8, 1.0, 1.2]]
    print(f"{str(c):40s} {re:6.3f} {ue:6.3f} {adm:6.3f} | " +
          " | ".join(f"{b:.4f} {u:.4f} {k:.3f}" for b, u, k in dyn), flush=True)
