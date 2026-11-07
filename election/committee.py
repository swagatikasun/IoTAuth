import argparse
import json
import math
import os
import numpy as np
from scipy.integrate import quad
from scipy.stats import binom, hypergeom

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def committee_f(k):
    return (k - 1) // 3


def hyper_tail(n, t, k):
    f = committee_f(k)
    if t <= f:
        return 0.0
    return float(hypergeom.sf(f, n, t, k))


def min_committee(n, t, eps, k_max=None):
    k_max = k_max or n
    for f in range(0, (k_max - 1) // 3 + 1):
        k = 3 * f + 1
        if k > n:
            break
        if hyper_tail(n, t, k) <= eps:
            return k
    return None


def group_counts(rho):
    levels, counts = np.unique(np.asarray(rho, dtype=float), return_counts=True)
    return levels, counts


def binomial_mix(levels, counts, theta, drop=None):
    dist = np.array([1.0])
    for lv, c in zip(levels, counts):
        if drop is not None and lv == drop:
            c = c - 1
        if c <= 0:
            continue
        p = 1.0 - theta / lv if theta < lv else 0.0
        b = binom.pmf(np.arange(c + 1), c, p)
        dist = np.convolve(dist, b)
    return dist


def weighted_tail(rho_byz, rho_hon, k):
    f = committee_f(k)
    lb, cb = group_counts(rho_byz) if len(rho_byz) else (np.array([]), np.array([], dtype=int))
    lh, ch = group_counts(rho_hon)
    t = int(np.sum(cb))

    def pick(dist, i):
        return dist[i] if 0 <= i < len(dist) else 0.0

    def density(theta):
        db = binomial_mix(lb, cb, theta)
        dh = binomial_mix(lh, ch, theta)
        total = np.zeros(k + 1)
        for lv, c in zip(lb, cb):
            if theta >= lv:
                continue
            dbr = binomial_mix(lb, cb, theta, drop=lv)
            for x in range(f + 1, min(k, t) + 1):
                total[x] += c / lv * pick(dbr, x - 1) * pick(dh, k - x)
        for lv, c in zip(lh, ch):
            if theta >= lv:
                continue
            dhr = binomial_mix(lh, ch, theta, drop=lv)
            for x in range(f + 1, min(k, t) + 1):
                total[x] += c / lv * pick(db, x) * pick(dhr, k - x - 1)
        return total.sum()

    breaks = sorted(set([0.0] + lb.tolist() + lh.tolist()))
    tail = 0.0
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        val, _ = quad(density, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-10)
        tail += val
    return float(tail)


def five_level_population(n, k, prev_leader=1):
    rest = n - k
    a = min(k, rest)
    b = min(k, rest - a)
    c = rest - a - b
    rho = [0.05] * prev_leader + [0.30] * (k - prev_leader) + [1.0] * a + [1.2] * b + [1.5] * c
    return np.array(rho)


def worst_case_split(rho, t):
    order = np.argsort(-rho, kind="stable")
    byz = rho[order[:t]]
    hon = rho[order[t:]]
    return byz, hon


def random_split_expectation(rho, t, k, samples=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(samples):
        perm = rng.permutation(len(rho))
        vals.append(weighted_tail(rho[perm[:t]], rho[perm[t:]], k))
    return float(np.mean(vals)), float(np.std(vals))


def two_level_population(n, k, rho_prev):
    return np.array([rho_prev] * k + [1.0] * (n - k))


def table_uniform():
    rows = []
    for n, k in [(100, 13), (100, 22), (100, 31), (100, 61), (1000, 13), (1000, 30), (1000, 31), (1000, 61), (1000, 100)]:
        for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            t = int(round(ratio * n))
            rows.append({"n": n, "K": k, "t_over_n": ratio, "t": t, "fail": hyper_tail(n, t, k)})
    minimum = []
    for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        t = int(round(ratio * 1000))
        for eps_name, eps in [("1e-3", 1e-3), ("1e-6", 1e-6), ("1e-9", 1e-9), ("2^-40", 2.0 ** -40)]:
            minimum.append({"t_over_n": ratio, "eps": eps_name, "K_min": min_committee(1000, t, eps)})
    return rows, minimum


def chernoff_old(k, ratio):
    gap = 1 / 3 - ratio
    return math.exp(-k * gap * gap / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["uniform", "weighted", "all"], default="all")
    args = ap.parse_args()
    out = {}
    if args.part in ("uniform", "all"):
        rows, minimum = table_uniform()
        out["uniform"] = rows
        out["min_committee"] = minimum
        out["old_bound_K30_K31"] = {"K30": chernoff_old(30, 0.3), "K31": chernoff_old(31, 0.3)}
        out["union_bound_example"] = {"E": 10 ** 6, "target": 1e-3, "per_epoch_eps": 1e-9,
                                      "K_min_t_over_n_0.10_n1000": min_committee(1000, 100, 1e-9)}
    if args.part in ("weighted", "all"):
        w = []
        for n, t in [(100, 30), (200, 60)]:
            k = 31
            rho = five_level_population(n, k)
            equal = np.ones(n)
            w.append({"n": n, "t": t, "rule": "equal multipliers (uniform)",
                      "fail": weighted_tail(equal[:t], equal[t:], k), "hypergeometric": hyper_tail(n, t, k)})
            byz, hon = worst_case_split(rho, t)
            w.append({"n": n, "t": t, "rule": "five-level, worst-case placement", "fail": weighted_tail(byz, hon, k)})
            for rp in [0.8, 0.5, 0.3]:
                r2 = two_level_population(n, k, rp)
                byz, hon = worst_case_split(r2, t)
                w.append({"n": n, "t": t, "rule": f"two-level rho_prev={rp}, worst case", "fail": weighted_tail(byz, hon, k)})
        out["weighted"] = w
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"committee_{args.part}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
