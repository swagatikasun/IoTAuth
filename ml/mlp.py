import argparse
import time
import numpy as np
from common import evaluate, make_split, mean_std, one_hot, partition_disjoint, save_json, sigmoid

HIDDEN = 32


def init(d, k, seed, hidden=HIDDEN):
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0, 1 / np.sqrt(d), (d, hidden)) * 0.5,
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, k)) * 0.5,
        "b2": np.zeros(k),
    }


def forward(p, x):
    a = x @ p["w1"] + p["b1"]
    h = a * a
    return a, h, h @ p["w2"] + p["b2"]


def grads(p, x, t):
    a, h, z = forward(p, x)
    e = (sigmoid(z) - t) / len(x)
    dh = e @ p["w2"].T
    da = dh * 2 * a
    return {"w1": x.T @ da, "b1": da.sum(0), "w2": h.T @ e, "b2": e.sum(0)}


def train_federated(xtr, ytr, k, clients, rounds, lr, beta, seed):
    p = init(xtr.shape[1], k, seed)
    v = {n: np.zeros_like(a) for n, a in p.items()}
    t = one_hot(ytr, k)
    n_total = len(ytr)
    for _ in range(rounds):
        agg = {n: np.zeros_like(a) for n, a in p.items()}
        for idx in clients:
            g = grads(p, xtr[idx], t[idx])
            for n in agg:
                agg[n] += g[n] * len(idx) / n_total
        for n in p:
            v[n] = beta * v[n] + lr * agg[n]
            p[n] = p[n] - v[n]
    return p


def as_linear_eval(p, x, y, k):
    _, _, z = forward(p, x)
    pred = z.argmax(1)
    return float(np.mean(pred == y))


def encrypted_inference(p, x, depth=4):
    from ckks_fedavg import Layout, make_context
    cc = make_context(depth=depth)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    layout = Layout(len(x), x.shape[1])
    cc.EvalRotateKeyGen(keys.secretKey, layout.segment_shifts())
    t0 = time.time()
    cts = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist())) for v in layout.pack_columns(x)]
    t_enc = time.time() - t0
    t0 = time.time()
    hidden = []
    for u in range(p["w1"].shape[1]):
        acc = None
        for b, ct in enumerate(cts):
            term = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(layout.weight_vector(p["w1"][:, u], b).tolist()))
            acc = term if acc is None else cc.EvalAdd(acc, term)
        for sh in layout.segment_shifts():
            acc = cc.EvalAdd(acc, cc.EvalRotate(acc, sh))
        acc = cc.EvalAdd(acc, float(p["b1"][u]))
        hidden.append(cc.EvalMult(acc, acc))
    outs = []
    for c in range(p["w2"].shape[1]):
        acc = None
        for u, h in enumerate(hidden):
            term = cc.EvalMult(h, float(p["w2"][u, c]))
            acc = term if acc is None else cc.EvalAdd(acc, term)
        outs.append(cc.EvalAdd(acc, float(p["b2"][c])))
    t_eval = time.time() - t0
    z = np.zeros((len(x), len(outs)))
    for c, ct in enumerate(outs):
        pt = cc.Decrypt(ct, keys.secretKey)
        pt.SetLength(len(x))
        z[:, c] = np.array(pt.GetRealPackedValue())[: len(x)]
    return z, t_enc, t_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--encrypted-seed", type=int, default=0)
    args = ap.parse_args()
    accs = []
    enc = None
    for seed in args.seeds:
        s = make_split(10, seed)
        k = s["k"]
        p = train_federated(s["xtr"], s["ytr"], k, partition_disjoint(s["ytr"], k), args.rounds, args.lr, args.beta, seed)
        acc = as_linear_eval(p, s["xte"], s["yte"], k)
        accs.append(acc)
        print(seed, "plaintext MLP accuracy %.4f" % acc, flush=True)
        if seed == args.encrypted_seed:
            z, t_enc, t_eval = encrypted_inference(p, s["xte"])
            _, _, zp = forward(p, s["xte"])
            enc = {
                "seed": seed,
                "test_samples": int(len(s["yte"])),
                "encrypted_accuracy": float(np.mean(z.argmax(1) == s["yte"])),
                "plaintext_accuracy": acc,
                "max_abs_logit_error": float(np.max(np.abs(z - zp))),
                "prediction_agreement": float(np.mean(z.argmax(1) == zp.argmax(1))),
                "encryption_seconds": t_enc,
                "inference_seconds": t_eval,
                "multiplicative_depth_used": 3,
            }
            print(enc, flush=True)
    save_json("mlp.json", {"hidden": HIDDEN, "rounds": args.rounds, "lr": args.lr, "beta": args.beta,
                           "plaintext_federated_accuracy": mean_std(accs), "encrypted_inference": enc})


if __name__ == "__main__":
    main()
