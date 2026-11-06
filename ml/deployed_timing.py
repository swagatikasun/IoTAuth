import argparse
import json
import os
import time
import numpy as np
import openfhe as fhe
import decision_params as dp

RESULTS = dp.RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", type=int, default=32768)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--scale", type=int, default=59)
    ap.add_argument("--first", type=int, default=60)
    ap.add_argument("--precision", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    dp.SHRINK = 36
    w = np.load(os.path.join(RESULTS, "ckks_weights_u10_s0_disjoint.npy"))
    d, k = w.shape
    rng = np.random.default_rng(1)
    x = rng.normal(size=d)
    x[-1] = 1.0
    cc1 = dp.context(fhe.ExecutionMode.EXEC_NOISE_ESTIMATION, None, args)
    kp1 = cc1.KeyGen()
    cc1.EvalMultKeyGen(kp1.secretKey)
    cc1.EvalRotateKeyGen(kp1.secretKey, dp.rotations(k))
    pt = cc1.Decrypt(dp.score_circuit(cc1, kp1, x, w, 3, 0.0, k), kp1.secretKey)
    noise = pt.GetLogError()
    cc = dp.context(fhe.ExecutionMode.EXEC_EVALUATION, noise, args)
    kp = cc.KeyGen()
    slots = cc.GetRingDimension() // 2
    f = np.zeros(slots)
    sel = np.zeros(slots)
    mk = np.zeros(slots)
    for c in range(k):
        f[c * dp.BLOCK: c * dp.BLOCK + d] = x
    sel[3 * dp.BLOCK] = 1.0
    mk[0] = float(rng.uniform(-(2.0 ** 43), 2.0 ** 43))
    times = []
    sizes = None
    for _ in range(args.repeats):
        t0 = time.time()
        cts = [cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist())) for v in (f, sel, mk)]
        times.append(time.time() - t0)
        if sizes is None:
            sizes = []
            for i, ct in enumerate(cts):
                path = f"/tmp/deployed_ct_{i}.bin"
                fhe.SerializeToFile(path, ct, fhe.BINARY)
                sizes.append(os.path.getsize(path))
                os.remove(path)
    out = {"args": vars(args), "log2_q": float(np.log2(cc.GetModulus())), "ring": cc.GetRingDimension(),
           "device_encrypt_seconds_mean": float(np.mean(times)), "device_encrypt_seconds": times,
           "request_ciphertext_bytes": sizes, "request_total_bytes": int(sum(sizes))}
    with open(os.path.join(RESULTS, "deployed_timing.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k2: out[k2] for k2 in ("log2_q", "device_encrypt_seconds_mean", "request_ciphertext_bytes", "request_total_bytes")}))


if __name__ == "__main__":
    main()
