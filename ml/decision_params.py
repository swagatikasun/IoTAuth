import argparse
import json
import os
import time
import numpy as np
import openfhe as fhe

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BLOCK = 256
SHRINK = 0


def context(mode, noise, args, multiparty=False):
    p = fhe.CCParamsCKKSRNS()
    p.SetDecryptionNoiseMode(fhe.DecryptionNoiseMode.NOISE_FLOODING_DECRYPT)
    p.SetExecutionMode(mode)
    if noise is not None:
        p.SetNoiseEstimate(noise)
        p.SetDesiredPrecision(args.precision)
    p.SetSecretKeyDist(fhe.SecretKeyDist.UNIFORM_TERNARY)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_NotSet)
    p.SetRingDim(args.ring)
    p.SetScalingTechnique(fhe.ScalingTechnique.FIXEDAUTO)
    p.SetScalingModSize(args.scale)
    p.SetFirstModSize(args.first)
    p.SetMultiplicativeDepth(args.depth)
    p.SetBatchSize(args.ring // 2)
    cc = fhe.GenCryptoContext(p)
    for f in (fhe.PKESchemeFeature.PKE, fhe.PKESchemeFeature.KEYSWITCH, fhe.PKESchemeFeature.LEVELEDSHE):
        cc.Enable(f)
    if multiparty:
        cc.Enable(fhe.PKESchemeFeature.MULTIPARTY)
    return cc


def score_circuit(cc, keys, x, w, u, m, k):
    slots = cc.GetRingDimension() // 2
    d = len(x)
    f = np.zeros(slots)
    wv = np.zeros(slots)
    sel = np.zeros(slots)
    mk = np.zeros(slots)
    for c in range(k):
        f[c * BLOCK: c * BLOCK + d] = x
        wv[c * BLOCK: c * BLOCK + d] = w[:, c]
    sel[u * BLOCK] = 1.0
    mk[0] = m
    enc = lambda v: cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist()))
    acc = cc.EvalMult(enc(f), cc.MakeCKKSPackedPlaintext(wv.tolist()))
    for i in range(int(np.log2(BLOCK))):
        acc = cc.EvalAdd(acc, cc.EvalRotate(acc, 1 << i))
    acc = cc.EvalMult(acc, enc(sel))
    for i in range(int(np.ceil(np.log2(k)))):
        acc = cc.EvalAdd(acc, cc.EvalRotate(acc, BLOCK << i))
    out = cc.EvalAdd(acc, enc(mk))
    return cc.EvalMult(out, 2.0 ** -SHRINK) if SHRINK else out


def rotations(k):
    return [1 << i for i in range(int(np.log2(BLOCK)))] + [BLOCK << i for i in range(int(np.ceil(np.log2(k))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", type=int, default=32768)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--scale", type=int, default=59)
    ap.add_argument("--first", type=int, default=60)
    ap.add_argument("--precision", type=int, default=80)
    ap.add_argument("--mask-bits", type=int, default=43)
    ap.add_argument("--parties", type=int, default=7)
    ap.add_argument("--shrink", type=int, default=36)
    args = ap.parse_args()
    global SHRINK
    SHRINK = args.shrink
    w = np.load(os.path.join(RESULTS, "ckks_weights_u10_s0_disjoint.npy"))
    rng = np.random.default_rng(0)
    d, k = w.shape
    x = rng.normal(size=d)
    x[-1] = 1.0
    s_true = float(x @ w[:, 3])
    out = {"args": vars(args), "flooding_statistical_security_bits": 30, "adversarial_queries": 1}

    cc1 = context(fhe.ExecutionMode.EXEC_NOISE_ESTIMATION, None, args)
    kp = cc1.KeyGen()
    cc1.EvalMultKeyGen(kp.secretKey)
    cc1.EvalRotateKeyGen(kp.secretKey, rotations(k))
    ct = score_circuit(cc1, kp, x, w, 3, 0.0, k)
    pt = cc1.Decrypt(ct, kp.secretKey)
    noise = pt.GetLogError()
    out["noise_estimate_log2"] = noise

    cc2 = context(fhe.ExecutionMode.EXEC_EVALUATION, noise, args)
    out["evaluation_log2_q"] = float(np.log2(cc2.GetModulus()))
    out["ring"] = cc2.GetRingDimension()
    kp2 = cc2.KeyGen()
    cc2.EvalMultKeyGen(kp2.secretKey)
    cc2.EvalRotateKeyGen(kp2.secretKey, rotations(k))
    m = float(rng.uniform(-(2.0 ** args.mask_bits), 2.0 ** args.mask_bits))
    t0 = time.time()
    ct2 = score_circuit(cc2, kp2, x, w, 3, m, k)
    out["evaluation_seconds"] = time.time() - t0
    pt2 = cc2.Decrypt(ct2, kp2.secretKey)
    pt2.SetLength(1)
    out["single_key_score_error"] = abs(pt2.GetRealPackedValue()[0] * 2.0 ** SHRINK - m - s_true)

    cc3 = context(fhe.ExecutionMode.EXEC_EVALUATION, noise, args, multiparty=True)
    kps = [cc3.KeyGen()]
    for _ in range(args.parties - 1):
        kps.append(cc3.MultipartyKeyGen(kps[-1].publicKey))
    v = np.zeros(cc3.GetRingDimension() // 2)
    v[0] = (s_true + m) * 2.0 ** -SHRINK
    c3 = cc3.Encrypt(kps[-1].publicKey, cc3.MakeCKKSPackedPlaintext(v.tolist()))
    for _ in range(args.depth):
        c3 = cc3.EvalMult(c3, 1.0)
    t0 = time.time()
    lead = cc3.MultipartyDecryptLead([c3], kps[0].secretKey)[0]
    mains = [cc3.MultipartyDecryptMain([c3], kp_.secretKey)[0] for kp_ in kps[1:]]
    out["partial_decrypt_seconds"] = (time.time() - t0) / args.parties
    fused = cc3.MultipartyDecryptFusion([lead] + mains)
    fused.SetLength(1)
    out["threshold_score_error"] = abs(fused.GetRealPackedValue()[0] * 2.0 ** SHRINK - m - s_true)
    out["threshold_log2_q"] = float(np.log2(cc3.GetModulus()))
    out["mask"] = m
    out["true_score"] = s_true
    print(json.dumps(out, indent=1))
    with open(os.path.join(RESULTS, "ckks128_headroom.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
