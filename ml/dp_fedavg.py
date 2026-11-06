import argparse
import numpy as np
from common import (Activation, BETA, GAMMA, ROUNDS, add_bias, evaluate, make_split, mean_std, one_hot,
                    partition_disjoint, save_json)


def rdp_epsilon(sigma, rounds, delta):
    alphas = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 512)])
    eps = rounds * alphas / (2 * sigma ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


def sigma_for(eps_target, rounds, delta):
    lo, hi = 0.05, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if rdp_epsilon(mid, rounds, delta) > eps_target:
            lo = mid
        else:
            hi = mid
    return hi


def train_dp(xtr, ytr, k, clients, sigma, clip, seed, rounds=ROUNDS, gamma=GAMMA, beta=BETA):
    rng = np.random.default_rng(seed)
    x = add_bias(xtr)
    t = one_hot(ytr, k)
    act = Activation("exact")
    w = np.zeros((x.shape[1], k))
    v = np.zeros_like(w)
    n = len(ytr)
    for _ in range(rounds):
        total = np.zeros_like(w)
        for idx in clients:
            xi, ti = x[idx], t[idx]
            e = act(xi @ w) - ti
            norms = np.linalg.norm(xi, axis=1) * np.linalg.norm(e, axis=1)
            scale = np.minimum(1.0, clip / np.maximum(norms, 1e-12))
            g = xi.T @ (e * scale[:, None])
            g = g + rng.normal(0, sigma * clip, g.shape)
            total += g
        v = beta * v + gamma * total / n
        w = w - v
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 4.0, 8.0])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()
    table = []
    for eps in args.eps:
        sigma = sigma_for(eps, ROUNDS, args.delta)
        accs = []
        for seed in args.seeds:
            s = make_split(10, seed)
            k = s["k"]
            w = train_dp(s["xtr"], s["ytr"], k, partition_disjoint(s["ytr"], k), sigma, args.clip, seed)
            accs.append(evaluate(w, s["xte"], s["yte"], k)["accuracy"])
        row = {"epsilon": eps, "delta": args.delta, "sigma": sigma, "clip": args.clip, "accuracy": mean_std(accs),
               "bytes_per_client_update": 8 * (s["xtr"].shape[1] + 1) * k}
        table.append(row)
        print(eps, "sigma %.2f acc %.2f+-%.2f" % (sigma, 100 * row["accuracy"][0], 100 * row["accuracy"][1]))
    save_json("dp_fedavg_clip%g.json" % args.clip, table)


if __name__ == "__main__":
    main()
