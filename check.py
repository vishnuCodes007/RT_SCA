import random, time
import simulate as S

rng = random.Random(1)
# 1) correctness of O(1) marginal and closed-form total risk vs brute force
for trial in range(50):
    batch = S.gen_batch(rng, 60, 0.4)
    H = S.hosts_for_load(batch, 0.7)
    p = S.Pool(H)
    S.alloc_ff(p, batch)
    s = S.gen_slice(rng, 0.4)
    for i in range(H * S.DOMAINS):
        a, b = p.marginal_agg(s, i), p.marginal_pairwise(s, i)
        assert abs(a - b) < 1e-9, (a, b)
    # brute-force total risk
    R = 0.0
    placed = [x for x in batch if x["loc"] is not None]
    for x in range(len(placed)):
        for y in range(x + 1, len(placed)):
            i, j = placed[x]["loc"], placed[y]["loc"]
            if i // S.DOMAINS != j // S.DOMAINS:
                continue
            lam = S.LAMBDA_C if i == j else S.LAMBDA_H
            R += lam * (placed[x]["sigma"] * placed[y]["tau"] + placed[y]["sigma"] * placed[x]["tau"])
    assert abs(R - p.total_risk()) < 1e-6, (R, p.total_risk())
print("marginal + closed-form risk: OK (50 trials)")

# 2) quick behaviour check at N=100, balanced, medium load
uf = 0.4
for name in S.METHODS:
    lats, res, uers = [], [], []
    rngw = random.Random(9)
    for t in range(8):
        batch = S.gen_batch(rngw, 100, uf)
        H = S.hosts_for_load(batch, 0.7)
        ref = S.Pool(H); S.alloc_ff(ref, S.clone(batch)); Rff = ref.total_risk()
        p = S.Pool(H)
        kw = {"K": 8, "ratio": S.forecast_ratio(uf)} if name == "RT-SCA" else {}
        r = S.run(name, p, S.clone(batch), random.Random(t), **kw)
        lats.append(r["lat"]); res.append(p.total_risk() / Rff); uers.append(p.uer())
    print(f"{name:8s} lat={sum(lats)/8:8.3f}ms  RE={sum(res)/8:.3f}  UER={sum(uers)/8:.3f}")

# 3) agg vs pairwise speed inside RT-SCA
batch = S.gen_batch(random.Random(3), 400, uf)
H = S.hosts_for_load(batch, 0.7)
for agg in (True, False):
    ts = []
    for t in range(10):
        p = S.Pool(H)
        r = S.run("RT-SCA", p, S.clone(batch), random.Random(t), K=8, ratio=S.forecast_ratio(uf), use_agg=agg)
        ts.append(r["lat"])
    print("use_agg", agg, f"{sum(ts)/10:.3f} ms")
