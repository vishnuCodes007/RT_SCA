"""Baselines on the SAME 20 instances used by the E4 ablation (seed SEED+99),
so the latency-security trade-off plot compares like with like."""
import json, random
import numpy as np
import simulate as S

uf = S.WORKLOADS["W2-balanced"]
rng_w = random.Random(S.SEED + 99)          # identical to exp_ablation
acc = {m: {"lat": [], "re": [], "uer": []} for m in ["FF", "ISO", "SCA-OPT"]}
for t in range(20):
    batch = S.gen_batch(rng_w, 100, uf)
    H = S.hosts_for_load(batch, 0.70)
    ref = S.Pool(H); S.alloc_ff(ref, S.clone(batch)); Rff = max(ref.total_risk(), 1e-9)
    for m in acc:
        pool = S.Pool(H)
        r = S.run(m, pool, S.clone(batch), random.Random(S.SEED + t))
        acc[m]["lat"].append(r["lat"]); acc[m]["re"].append(pool.total_risk() / Rff)
        acc[m]["uer"].append(pool.uer())
out = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
res = json.load(open("results.json"))
res["E4ref"] = out
json.dump(res, open("results.json", "w"), indent=2)
print(json.dumps(out, indent=1))
