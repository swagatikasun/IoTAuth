import argparse
import os
import numpy as np
from common import (Activation, RESULTS, add_bias, evaluate, make_split, mean_std, partition_disjoint,
                    save_json, sigmoid, train_fedavg)


class NoClip:
    def __init__(self, kind, a):
        self.inner = Activation(kind, a=a)
        self.kind = kind

    def __call__(self, z):
        return np.polyval(self.inner.coef[::-1], z)


def approximation_error(act, lo, hi):
    z = np.linspace(lo, hi, 4001)
    return float(np.max(np.abs(act(z) - sigmoid(z))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--intervals", type=float, nargs="+", default=[4.0, 8.0, 12.0])
    args = ap.parse_args()
    variants = [("exact sigmoid (plaintext)", Activation("exact"), None)]
    variants.append(("linear 0.5+z/4", Activation("linear"), 1))
    for a in args.intervals:
        for kind, label in (("ls3", "least squares"), ("minimax3", "minimax")):
            variants.append((f"degree-3 {label}, a={a:g}, clipped (plaintext only)", Activation(kind, a=a), 2))
            variants.append((f"degree-3 {label}, a={a:g}, unclipped (as in CKKS)", NoClip(kind, a), 2))
    rows = {}
    ranges = {}
    diverged = {}
    for seed in args.seeds:
        s = make_split(10, seed)
        k = s["k"]
        clients = partition_disjoint(s["ytr"], k)
        for name, act, _ in variants:
            hist = []
            with np.errstate(all="ignore"):
                w = train_fedavg(s["xtr"], s["ytr"], k, clients, act=act, history=hist, xte=s["xte"], yte=s["yte"])
            accs = [h["accuracy"] for h in hist]
            rows.setdefault(name, []).append(evaluate(w, s["xte"], s["yte"], k)["accuracy"] if np.all(np.isfinite(w)) else 0.1)
            bad = [i + 1 for i, a_ in enumerate(accs) if a_ <= 0.2]
            diverged.setdefault(name, []).append(bad[0] if bad else None)
            if not np.all(np.isfinite(w)):
                ranges.setdefault(name, []).append([np.nan] * 4)
                continue
            z = add_bias(s["xtr"]) @ w
            ranges.setdefault(name, []).append([float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5)),
                                                float(z.min()), float(z.max())])
    table = []
    for name, act, extra_depth in variants:
        r = np.array(ranges[name], dtype=float)
        lo, hi = float(np.nanmin(r[:, 0])) if np.isfinite(r[:, 0]).any() else np.nan, float(np.nanmax(r[:, 1])) if np.isfinite(r[:, 1]).any() else np.nan
        row = {
            "variant": name,
            "accuracy": mean_std(rows[name]),
            "extra_depth_for_activation": extra_depth,
            "logit_range_p0.5_p99.5": [lo, hi],
            "diverged_at_round": diverged[name],
        }
        if extra_depth is not None and np.isfinite(lo):
            row["max_abs_error_on_observed_range"] = approximation_error(act, lo, hi)
        table.append(row)
        print(name, "acc %.2f+-%.2f" % (100 * row["accuracy"][0], 100 * row["accuracy"][1]),
              "range [%.2f, %.2f]" % (lo, hi), row.get("max_abs_error_on_observed_range"), "diverged", diverged[name])
    coef = {}
    for a in args.intervals:
        coef[f"ls3_a{a:g}"] = Activation("ls3", a=a).coef
        coef[f"minimax3_a{a:g}"] = Activation("minimax3", a=a).coef
    save_json("sigmoid_ablation.json", {"table": table, "coefficients": coef})


if __name__ == "__main__":
    main()
