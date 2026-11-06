import argparse
import os
import numpy as np
from common import (Activation, RESULTS, add_bias, confusion, evaluate, make_split, mean_std,
                    partition_disjoint, save_json, train_centralized, train_fedavg)
from plots import bar_with_error, confusion_panels


def ckks_weights(seed):
    path = os.path.join(RESULTS, f"ckks_weights_u10_s{seed}_disjoint.npy")
    return np.load(path) if os.path.exists(path) else None


def run_seed(seed):
    s = make_split(10, seed)
    k = s["k"]
    clients = partition_disjoint(s["ytr"], k)
    exact, linear = Activation("exact"), Activation("linear")
    models = {
        "centralized": train_centralized(s["xtr"], s["ytr"], k, act=exact),
        "fedavg": train_fedavg(s["xtr"], s["ytr"], k, clients, act=exact),
    }
    emulated = train_fedavg(s["xtr"], s["ytr"], k, clients, act=linear)
    enc = ckks_weights(seed)
    models["fedavg_he"] = enc if enc is not None else emulated
    out = {"seed": seed, "he_source": "ckks" if enc is not None else "plaintext-emulation"}
    for name, w in models.items():
        out[name] = evaluate(w, s["xte"], s["yte"], k)
    logits = add_bias(s["xtr"]) @ models["fedavg_he"]
    out["he_logit_range"] = [float(logits.min()), float(logits.max())]
    out["he_logit_p01_p99"] = [float(np.percentile(logits, 1)), float(np.percentile(logits, 99))]
    if enc is not None:
        out["ckks_vs_emulation_max_abs_diff"] = float(np.abs(enc - emulated).max())
    return out, models, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    runs = []
    for seed in args.seeds:
        out, models, s = run_seed(seed)
        runs.append(out)
        print(seed, out["he_source"], {n: round(out[n]["accuracy"], 4) for n in ("centralized", "fedavg", "fedavg_he")})
        if seed == args.seeds[0]:
            mats = {n: confusion(w, s["xte"], s["yte"], s["k"]) for n, w in models.items()}
            save_json("confusion_seed%d.json" % seed, mats)
            confusion_panels(mats, os.path.join(RESULTS, "fig3_confusion.pdf"))
    summary = {}
    for name in ("centralized", "fedavg", "fedavg_he"):
        summary[name] = {}
        for metric in ("accuracy", "macro_recall", "macro_precision", "min_recall", "std_recall"):
            m, sd = mean_std([r[name][metric] for r in runs])
            summary[name][metric] = {"mean": m, "std": sd}
    save_json("main_results.json", {"runs": runs, "summary": summary})
    bar_with_error(summary, os.path.join(RESULTS, "fig4_accuracy.pdf"))
    for name, v in summary.items():
        print(name, {k: "%.2f +- %.2f" % (100 * x["mean"], 100 * x["std"]) for k, x in v.items()})


if __name__ == "__main__":
    main()
