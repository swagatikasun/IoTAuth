import argparse
import json
import os
import time
import numpy as np
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature, SecurityLevel)
from common import (Activation, GAMMA, MU, RESULTS, add_bias, evaluate, make_split, one_hot,
                    partition_disjoint, partition_dirichlet, partition_groups, partition_iid, save_json)

SLOTS = 16384


def make_context(depth=4, scale_bits=50, first_bits=60, ring=32768, digits=0):
    p = CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(scale_bits)
    p.SetFirstModSize(first_bits)
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    p.SetRingDim(ring)
    p.SetBatchSize(ring // 2)
    if digits:
        p.SetNumLargeDigits(digits)
    cc = GenCryptoContext(p)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE):
        cc.Enable(feat)
    return cc


class Layout:
    def __init__(self, n_samples, n_features, slots=SLOTS):
        seg = 1
        while seg < n_samples:
            seg *= 2
        self.seg = seg
        self.per_ct = slots // seg
        self.n_ct = -(-n_features // self.per_ct)
        self.slots = slots
        self.n_features = n_features

    def feature_index(self, b, s):
        return b * self.per_ct + s

    def segment_shifts(self):
        out, k = [], self.seg
        while k < self.slots:
            out.append(k)
            k *= 2
        return out

    def sample_shifts(self):
        out, k = [], 1
        while k < self.seg:
            out.append(k)
            k *= 2
        return out

    def pack_columns(self, x):
        cts = []
        for b in range(self.n_ct):
            v = np.zeros(self.slots)
            for s in range(self.per_ct):
                j = self.feature_index(b, s)
                if j < x.shape[1]:
                    v[s * self.seg: s * self.seg + x.shape[0]] = x[:, j]
            cts.append(v)
        return cts

    def replicate_samples(self, col):
        v = np.zeros(self.slots)
        for s in range(self.per_ct):
            v[s * self.seg: s * self.seg + len(col)] = col
        return v

    def weight_vector(self, w_col, b):
        v = np.zeros(self.slots)
        for s in range(self.per_ct):
            j = self.feature_index(b, s)
            if j < len(w_col):
                v[s * self.seg:(s + 1) * self.seg] = w_col[j]
        return v

    def head_mask(self, scale):
        v = np.zeros(self.slots)
        v[np.arange(self.per_ct) * self.seg] = scale
        return v

    def unpack_heads(self, decoded, b):
        out = {}
        for s in range(self.per_ct):
            j = self.feature_index(b, s)
            if j < self.n_features:
                out[j] = decoded[s * self.seg]
        return out


class EncryptedFedAvg:
    def __init__(self, cc, keys, layout, k):
        self.cc = cc
        self.keys = keys
        self.layout = layout
        self.k = k
        self.timing = {"client_eval": 0.0, "reduce": 0.0, "decrypt": 0.0}

    def encrypt(self, vec):
        return self.cc.Encrypt(self.keys.publicKey, self.cc.MakeCKKSPackedPlaintext(vec.tolist()))

    def pt(self, vec):
        return self.cc.MakeCKKSPackedPlaintext(vec.tolist())

    def rotsum(self, ct, shifts):
        acc = ct
        for sh in shifts:
            acc = self.cc.EvalAdd(acc, self.cc.EvalRotate(acc, sh))
        return acc

    def encrypt_client(self, x, y, c0=0.5):
        xs = [self.encrypt(v) for v in self.layout.pack_columns(add_bias(x))]
        t = one_hot(y, self.k)
        ys = [self.encrypt(self.layout.replicate_samples(c0 - t[:, c])) for c in range(self.k)]
        valid = self.layout.replicate_samples(np.ones(len(y)))
        return {"x": xs, "y": ys, "n": len(y), "valid": valid}

    def activation(self, z, coef):
        if coef is None:
            return z
        z2 = self.cc.EvalMult(z, z)
        z3 = self.cc.EvalMult(z2, z)
        out = self.cc.EvalMult(z, float(coef[1]))
        out = self.cc.EvalAdd(out, self.cc.EvalMult(z2, float(coef[2])))
        return self.cc.EvalAdd(out, self.cc.EvalMult(z3, float(coef[3])))

    def client_products(self, client, w, mu, coef=None):
        start = time.time()
        prods = []
        seg_shifts = self.layout.segment_shifts()
        for c in range(self.k):
            z = None
            for b, ct in enumerate(client["x"]):
                term = self.cc.EvalMult(ct, self.pt(mu * self.layout.weight_vector(w[:, c], b)))
                z = term if z is None else self.cc.EvalAdd(z, term)
            z = self.rotsum(z, seg_shifts)
            e = self.cc.EvalAdd(self.activation(z, coef), client["y"][c])
            prods.append([self.cc.EvalMult(e, ct) for ct in client["x"]])
        self.timing["client_eval"] += time.time() - start
        return prods

    def reduce(self, prods, n_total):
        start = time.time()
        mask = self.pt(self.layout.head_mask(1.0 / n_total))
        out = []
        for c in range(self.k):
            row = []
            for ct in prods[c]:
                s = self.rotsum(ct, self.layout.sample_shifts())
                row.append(self.cc.EvalMult(s, mask))
            out.append(row)
        self.timing["reduce"] += time.time() - start
        return out

    def pack(self, reduced):
        packed = None
        for c, row in enumerate(reduced):
            inner = None
            for b, ct in enumerate(row):
                r = ct if b == 0 else self.cc.EvalRotate(ct, -b)
                inner = r if inner is None else self.cc.EvalAdd(inner, r)
            shifted = inner if c == 0 else self.cc.EvalRotate(inner, -self.layout.n_ct * c)
            packed = shifted if packed is None else self.cc.EvalAdd(packed, shifted)
        return packed

    def pack_shifts(self):
        return [-b for b in range(1, self.layout.n_ct)] + [-self.layout.n_ct * c for c in range(1, self.k)]

    def decrypt_packed(self, ct, d):
        start = time.time()
        p = self.cc.Decrypt(ct, self.keys.secretKey)
        p.SetLength(self.layout.slots)
        vals = np.array(p.GetRealPackedValue())
        g = np.zeros((d, self.k))
        for c in range(self.k):
            for b in range(self.layout.n_ct):
                for s_ in range(self.layout.per_ct):
                    j = self.layout.feature_index(b, s_)
                    if j < d:
                        g[j, c] = vals[s_ * self.layout.seg + c * self.layout.n_ct + b]
        self.timing["decrypt"] += time.time() - start
        return g

    def decrypt_gradient(self, enc, d):
        start = time.time()
        g = np.zeros((d, self.k))
        for c in range(self.k):
            for b, ct in enumerate(enc[c]):
                p = self.cc.Decrypt(ct, self.keys.secretKey)
                p.SetLength(self.layout.slots)
                vals = np.array(p.GetRealPackedValue())
                for j, v in self.layout.unpack_heads(vals, b).items():
                    g[j, c] = v
        self.timing["decrypt"] += time.time() - start
        return g

    def add_into(self, acc, prods):
        if acc is None:
            return prods
        return [[self.cc.EvalAdd(a, p) for a, p in zip(ra, rp)] for ra, rp in zip(acc, prods)]


class PolyNoClip:
    def __init__(self, act):
        self.coef = act.coef

    def __call__(self, z):
        return np.polyval(self.coef[::-1], z)


def ct_size(ct):
    import tempfile
    from openfhe import BINARY, SerializeToFile
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    SerializeToFile(path, ct, BINARY)
    size = os.path.getsize(path)
    os.remove(path)
    return size


def plaintext_reference(xtr, ytr, k, clients, rounds, mu, gamma, beta, act=None):
    x = add_bias(xtr)
    t = one_hot(ytr, k)
    act = act or Activation("linear", mu=mu)
    w = np.zeros((x.shape[1], k))
    v = np.zeros_like(w)
    n = len(ytr)
    for _ in range(rounds):
        g = sum(x[idx].T @ (act(x[idx] @ w) - t[idx]) for idx in clients) / n
        v = beta * v + g
        w = w - gamma * v
    return w


def run(args):
    s = make_split(args.users, args.seed)
    k = s["k"]
    if args.partition == "disjoint":
        clients = partition_disjoint(s["ytr"], k, args.users_per_client)
    elif args.partition == "groups":
        clients = partition_groups(s["ytr"], k, args.clients)
    elif args.partition == "iid":
        clients = partition_iid(s["ytr"], args.clients, args.seed)
    else:
        clients = partition_dirichlet(s["ytr"], args.clients, args.alpha, args.seed)
    d = s["xtr"].shape[1] + 1
    layout = Layout(max(len(c) for c in clients), d)
    cc = make_context(depth=args.depth, ring=args.ring)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    fed = EncryptedFedAvg(cc, keys, layout, k)
    shifts = layout.segment_shifts() + layout.sample_shifts()
    if args.reduce == "client":
        shifts += fed.pack_shifts()
    cc.EvalRotateKeyGen(keys.secretKey, shifts)
    if args.act == "linear":
        act, coef, mu, c0 = Activation("linear", mu=args.mu), None, args.mu, 0.5
    else:
        act = Activation(args.act, a=args.a)
        coef, mu, c0 = act.coef, 1.0, float(act.coef[0])
    t0 = time.time()
    enc_clients = [fed.encrypt_client(s["xtr"][idx], s["ytr"][idx], c0) for idx in clients]
    enc_time = time.time() - t0
    tag = f"u{args.users}_s{args.seed}_{args.partition}" + ("" if args.act == "linear" else f"_{args.act}_a{args.a:g}") + ("_clientreduce" if args.reduce == "client" else "") + args.tag_suffix
    state_path = os.path.join(RESULTS, f"ckks_state_{tag}.npz")
    w = np.zeros((d, k))
    v = np.zeros_like(w)
    start_round = 0
    log = []
    if args.resume and os.path.exists(state_path):
        st = np.load(state_path, allow_pickle=True)
        w, v, start_round = st["w"], st["v"], int(st["round"])
        log = list(st["log"])
    n_total = len(s["ytr"])
    end_round = min(args.rounds, start_round + args.batch_rounds)
    for r in range(start_round, end_round):
        rs = time.time()
        if args.reduce == "client":
            total = None
            for client in enc_clients:
                t0 = time.time()
                packed = fed.pack(fed.reduce(fed.client_products(client, w, mu, coef), n_total))
                fed.timing.setdefault("per_client_seconds", []).append(time.time() - t0)
                fed.timing["upload_bytes_per_client"] = ct_size(packed)
                total = packed if total is None else cc.EvalAdd(total, packed)
            g = fed.decrypt_packed(total, d)
        else:
            acc = None
            for client in enc_clients:
                prods = fed.client_products(client, w, mu, coef)
                if "upload_bytes_per_client" not in fed.timing:
                    fed.timing["upload_bytes_per_client"] = sum(ct_size(ct) for row in prods for ct in row)
                acc = fed.add_into(acc, prods)
            g = fed.decrypt_gradient(fed.reduce(acc, n_total), d)
        v = args.beta * v + g
        w = w - args.gamma * v
        ev = evaluate(w, s["xte"], s["yte"], k)
        log.append({"round": r + 1, "seconds": time.time() - rs, "accuracy": ev["accuracy"]})
        print(log[-1], flush=True)
        np.savez(state_path, w=w, v=v, round=r + 1, log=np.array(log, dtype=object))
    ref = plaintext_reference(s["xtr"], s["ytr"], k, clients, end_round, args.mu, args.gamma, args.beta, act=None if args.act == "linear" else PolyNoClip(act))
    out = {
        "config": vars(args),
        "ring": cc.GetRingDimension(),
        "log2_q": float(np.log2(cc.GetModulus())),
        "layout": {"segment": layout.seg, "features_per_ct": layout.per_ct, "cts_per_client": layout.n_ct},
        "client_encryption_seconds": enc_time,
        "rounds_done": end_round,
        "timing_totals": fed.timing,
        "log": log,
        "final_encrypted": evaluate(w, s["xte"], s["yte"], k),
        "final_plaintext_reference": evaluate(ref, s["xte"], s["yte"], k),
        "max_abs_weight_diff": float(np.max(np.abs(w - ref))),
        "prediction_agreement": float(np.mean(np.argmax(add_bias(s["xte"]) @ w, 1) == np.argmax(add_bias(s["xte"]) @ ref, 1))),
    }
    np.save(os.path.join(RESULTS, f"ckks_weights_{tag}.npy"), w)
    save_json(f"ckks_fedavg_{tag}.json", out)
    print(json.dumps({k2: out[k2] for k2 in ("rounds_done", "final_encrypted", "final_plaintext_reference", "max_abs_weight_diff", "prediction_agreement")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--batch-rounds", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--partition", choices=["disjoint", "groups", "iid", "dirichlet"], default="disjoint")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--users-per-client", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--act", choices=["linear", "ls3", "minimax3"], default="linear")
    ap.add_argument("--a", type=float, default=8.0)
    ap.add_argument("--reduce", choices=["aggregator", "client"], default="aggregator")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ring", type=int, default=32768)
    ap.add_argument("--tag-suffix", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
