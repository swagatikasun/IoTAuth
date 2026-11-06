import argparse
import numpy as np
from scipy.optimize import minimize
from common import Activation, add_bias, make_split, mean_std, one_hot, partition_disjoint, save_json, sigmoid, train_fedavg


def observed_gradient(w, x, t, act):
    xa = add_bias(x)
    return xa.T @ (act(xa @ w) - t) / len(x)


def analytic_single(g):
    for c in range(g.shape[1]):
        if abs(g[-1, c]) > 1e-12:
            return g[:-1, c] / g[-1, c]
    return np.zeros(g.shape[0] - 1)


def optimisation_attack(g, w, t, act_kind, d, b, seed, restarts=3):
    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf

    def loss_grad(flat):
        a = np.hstack([flat.reshape(b, d), np.ones((b, 1))])
        z = a @ w
        if act_kind == "linear":
            p, sp = 0.5 + 0.25 * z, 0.25 * np.ones_like(z)
        else:
            p = sigmoid(z)
            sp = p * (1 - p)
        e = p - t
        r = a.T @ e / b - g
        grad = 2.0 / b * (e @ r.T + ((a @ r) * sp) @ w.T)
        return float(np.sum(r * r)), grad[:, :d].ravel()

    for _ in range(restarts):
        x0 = rng.normal(0, 1, b * d)
        res = minimize(loss_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    return best.reshape(b, d)


def match_score(x_true, x_hat):
    b = len(x_true)
    used, mses, cors = set(), [], []
    for i in range(b):
        cands = [(np.mean((x_true[i] - x_hat[j]) ** 2), j) for j in range(b) if j not in used]
        m, j = min(cands)
        used.add(j)
        mses.append(m)
        cors.append(np.corrcoef(x_true[i], x_hat[j])[0, 1])
    return float(np.mean(mses)), float(np.mean(cors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    out = {}
    for seed in args.seeds:
        s = make_split(10, seed)
        k = s["k"]
        d = s["xtr"].shape[1]
        rng = np.random.default_rng(seed)
        w = train_fedavg(s["xtr"], s["ytr"], k, partition_disjoint(s["ytr"], k), rounds=10, act=Activation("exact"))
        prior = s["xtr"].mean(axis=0)
        for b in args.batches:
            for kind in ("exact", "linear"):
                act = Activation(kind)
                for trial in range(args.trials):
                    pick = rng.choice(len(s["ytr"]), b, replace=False)
                    x, t = s["xtr"][pick], one_hot(s["ytr"][pick], k)
                    g = observed_gradient(w, x, t, act)
                    if b == 1:
                        xh = analytic_single(g)[None, :]
                        method = "analytic"
                    else:
                        xh = optimisation_attack(g, w, t, kind, d, b, seed * 100 + trial)
                        method = "optimisation"
                    mse, cor = match_score(x, xh)
                    pm, pc = match_score(x, np.tile(prior, (b, 1)) + 1e-9 * rng.normal(size=(b, d)))
                    out.setdefault((b, kind, method), []).append((mse, cor, pm, pc))
        print("seed", seed, "done", flush=True)
    table = []
    for (b, kind, method), vals in out.items():
        v = np.array(vals)
        row = {"batch": b, "activation": kind, "attack": method,
               "plaintext_mse": mean_std(v[:, 0]), "plaintext_corr": mean_std(v[:, 1]),
               "ciphertext_only_mse": mean_std(v[:, 2]), "ciphertext_only_corr": mean_std(v[:, 3])}
        table.append(row)
        print(b, kind, method, "plain mse %.3g corr %.3f | ct-only mse %.3g corr %.3f" % (
            row["plaintext_mse"][0], row["plaintext_corr"][0], row["ciphertext_only_mse"][0], row["ciphertext_only_corr"][0]))
    save_json("inversion.json", table)


if __name__ == "__main__":
    main()
