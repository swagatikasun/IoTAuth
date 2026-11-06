import argparse
import os
import tempfile
import time
import numpy as np
from openfhe import BINARY, SerializeToFile
from ckks_fedavg import make_context
from common import RESULTS, add_bias, make_split, save_json

BLOCK = 256


def ct_bytes(ct):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    SerializeToFile(path, ct, BINARY)
    size = os.path.getsize(path)
    os.remove(path)
    return size


class RequestCodec:
    def __init__(self, cc, keys, k, d):
        self.cc = cc
        self.keys = keys
        self.k = k
        self.d = d
        self.slots = cc.GetRingDimension() // 2

    def feature_vector(self, x):
        v = np.zeros(self.slots)
        for c in range(self.k):
            v[c * BLOCK: c * BLOCK + self.d] = x
        return v

    def selector_vector(self, u):
        v = np.zeros(self.slots)
        v[u * BLOCK] = 1.0
        return v

    def mask_vector(self, m):
        v = np.zeros(self.slots)
        v[0] = m
        return v

    def encrypt_request(self, x, u, m):
        enc = lambda v: self.cc.Encrypt(self.keys.publicKey, self.cc.MakeCKKSPackedPlaintext(v.tolist()))
        return enc(self.feature_vector(x)), enc(self.selector_vector(u)), enc(self.mask_vector(m))

    def model_plaintext(self, w):
        v = np.zeros(self.slots)
        for c in range(self.k):
            v[c * BLOCK: c * BLOCK + self.d] = w[:, c]
        return self.cc.MakeCKKSPackedPlaintext(v.tolist())

    def rotations(self):
        inner = [1 << i for i in range(int(np.log2(BLOCK)))]
        outer = [BLOCK << i for i in range(int(np.ceil(np.log2(self.k))))]
        return inner + outer

    def evaluate(self, ctf, cts, ctm, wpt):
        cc = self.cc
        acc = cc.EvalMult(ctf, wpt)
        for sh in [1 << i for i in range(int(np.log2(BLOCK)))]:
            acc = cc.EvalAdd(acc, cc.EvalRotate(acc, sh))
        acc = cc.EvalMult(acc, cts)
        for sh in [BLOCK << i for i in range(int(np.ceil(np.log2(self.k))))]:
            acc = cc.EvalAdd(acc, cc.EvalRotate(acc, sh))
        return cc.EvalAdd(acc, ctm)


def bench_requests(args, w, s):
    cc = make_context(depth=args.depth, scale_bits=args.scale_bits, first_bits=args.first_bits)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    k = s["k"]
    d = s["xtr"].shape[1] + 1
    codec = RequestCodec(cc, keys, k, d)
    cc.EvalRotateKeyGen(keys.secretKey, codec.rotations())
    wpt = codec.model_plaintext(w)
    rng = np.random.default_rng(0)
    xs = add_bias(s["xte"])
    plain_scores = xs @ w
    enc_times, sizes = [], None
    out = {"ring": cc.GetRingDimension(), "log2_q": float(np.log2(cc.GetModulus())), "depth": args.depth,
           "scale_bits": args.scale_bits, "first_bits": args.first_bits}
    for i in range(20):
        t0 = time.time()
        cts = codec.encrypt_request(xs[i], int(s["yte"][i]), 0.0)
        enc_times.append(time.time() - t0)
        if sizes is None:
            sizes = [ct_bytes(c) for c in cts]
    out["device_encrypt_seconds_mean"] = float(np.mean(enc_times))
    out["request_ciphertext_bytes"] = sizes
    out["request_total_bytes"] = int(sum(sizes))
    batches = {}
    errors = []
    for m in args.batch_sizes:
        t_eval, t_dec, recovered, last = 0.0, 0.0, [], None
        masks = rng.uniform(-args.mask_range, args.mask_range, m)
        idx = rng.choice(len(xs), m, replace=False)
        for j, i in enumerate(idx):
            f, sel, mk = codec.encrypt_request(xs[i], int(s["yte"][i]), masks[j])
            t0 = time.time()
            last = codec.evaluate(f, sel, mk, wpt)
            t_eval += time.time() - t0
            t0 = time.time()
            pt = cc.Decrypt(last, keys.secretKey)
            pt.SetLength(1)
            recovered.append(pt.GetRealPackedValue()[0] - masks[j])
            t_dec += time.time() - t0
        truth = plain_scores[idx, s["yte"][idx]]
        err = np.abs(np.array(recovered) - truth)
        errors.extend(err.tolist())
        batches[m] = {"evaluation_seconds": t_eval, "per_request_seconds": t_eval / m, "single_key_decrypt_seconds": t_dec,
                      "max_abs_score_error": float(err.max())}
        print("M", m, batches[m], flush=True)
    out["batches"] = batches
    out["result_ciphertext_bytes"] = ct_bytes(last)
    out["mask_range"] = args.mask_range
    out["max_abs_score_error"] = float(np.max(errors))
    return out


def bench_mask_headroom(args, s, w):
    rows = []
    xs = add_bias(s["xte"][:5])
    true = (xs @ w)[np.arange(5), s["yte"][:5]]
    for scale_bits, first_bits in [(50, 60), (40, 60), (30, 60), (25, 60)]:
        cc = make_context(depth=1, scale_bits=scale_bits, first_bits=first_bits)
        keys = cc.KeyGen()
        for mask_bits in (8, 16, 24, 32, 40, 44):
            rng = np.random.default_rng(mask_bits)
            m = rng.uniform(-(2.0 ** mask_bits), 2.0 ** mask_bits, 5)
            v = np.zeros(cc.GetRingDimension() // 2)
            v[:5] = true + m
            try:
                ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist()))
                ct = cc.EvalMult(ct, 1.0)
                pt = cc.Decrypt(ct, keys.secretKey)
                pt.SetLength(5)
                rec = np.array(pt.GetRealPackedValue()[:5]) - m
                err = float(np.max(np.abs(rec - true)))
            except Exception as e:
                err = "failed: " + str(e).splitlines()[-1][-80:]
            rows.append({"scale_bits": scale_bits, "first_modulus_bits": first_bits, "mask_bits": mask_bits, "max_abs_error": err})
            print(rows[-1], flush=True)
    return rows


def bench_threshold(args):
    from openfhe import PKESchemeFeature
    rows = []
    for parties in args.parties:
        cc = make_context(depth=args.depth)
        cc.Enable(PKESchemeFeature.MULTIPARTY)
        kps = [cc.KeyGen()]
        for _ in range(parties - 1):
            kps.append(cc.MultipartyKeyGen(kps[-1].publicKey))
        joint = kps[-1].publicKey
        v = np.random.default_rng(0).normal(size=cc.GetRingDimension() // 2)
        ct = cc.Encrypt(joint, cc.MakeCKKSPackedPlaintext(v.tolist()))
        for _ in range(args.depth - 1):
            ct = cc.EvalMult(ct, 1.0)
        t0 = time.time()
        shares = cc.MultipartyDecryptLead([ct], kps[0].secretKey)
        t_lead = time.time() - t0
        mains = []
        t0 = time.time()
        for kp in kps[1:]:
            mains.append(cc.MultipartyDecryptMain([ct], kp.secretKey)[0])
        t_main = (time.time() - t0) / max(1, parties - 1)
        t0 = time.time()
        pt = cc.MultipartyDecryptFusion([shares[0]] + mains)
        t_fuse = time.time() - t0
        pt.SetLength(8)
        err = float(np.max(np.abs(np.array(pt.GetRealPackedValue()[:8]) - v[:8])))
        rows.append({"parties": parties, "partial_decrypt_seconds": t_main, "lead_seconds": t_lead,
                     "fusion_seconds": t_fuse, "share_bytes": ct_bytes(mains[0]) if mains else ct_bytes(shares[0]),
                     "decryption_error": err})
        print(rows[-1], flush=True)
    return rows


def bench_wrong_key(s):
    cc = make_context(depth=2)
    owner = cc.KeyGen()
    attacker = cc.KeyGen()
    x = add_bias(s["xtr"][:1])[0]
    v = np.zeros(cc.GetRingDimension() // 2)
    v[: len(x)] = x
    ct = cc.Encrypt(owner.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist()))
    try:
        pt = cc.Decrypt(ct, attacker.secretKey)
        pt.SetLength(len(x))
        guess = np.array(pt.GetRealPackedValue()[: len(x)])
        return {"outcome": "decoded", "correlation_with_true_features": float(np.corrcoef(guess, x)[0, 1])}
    except Exception as e:
        return {"outcome": "decryption rejected by library", "message": str(e).splitlines()[-1][-160:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["requests", "headroom", "threshold", "wrongkey", "all"], default="all")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[10, 50, 100, 500])
    ap.add_argument("--parties", type=int, nargs="+", default=[4, 7, 13, 31])
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--scale-bits", type=int, default=50)
    ap.add_argument("--first-bits", type=int, default=60)
    ap.add_argument("--mask-range", type=float, default=2.0 ** 8)
    args = ap.parse_args()
    s = make_split(10, 0)
    wpath = os.path.join(RESULTS, "ckks_weights_u10_s0_disjoint.npy")
    if os.path.exists(wpath):
        w = np.load(wpath)
    else:
        from common import Activation, partition_disjoint, train_fedavg
        w = train_fedavg(s["xtr"], s["ytr"], s["k"], partition_disjoint(s["ytr"], s["k"]), act=Activation("linear"))
    out = {}
    if args.part in ("requests", "all"):
        out["requests"] = bench_requests(args, w, s)
    if args.part in ("headroom", "all"):
        out["mask_headroom"] = bench_mask_headroom(args, s, w)
    if args.part in ("threshold", "all"):
        out["threshold"] = bench_threshold(args)
    if args.part in ("wrongkey", "all"):
        out["wrong_key"] = bench_wrong_key(s)
        print(out["wrong_key"])
    save_json(f"ckks_bench_{args.part}.json", out)


if __name__ == "__main__":
    main()
