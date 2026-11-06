import argparse
import os
from common import (Activation, RESULTS, evaluate, make_split, mean_std, partition_dirichlet,
                    partition_disjoint, partition_iid, save_json, train_centralized, train_fedavg)
from plots import curves


def partitions(ytr, k, seed):
    return {
        "IID": partition_iid(ytr, 5, seed),
        "Dirichlet a=1.0": partition_dirichlet(ytr, 5, 1.0, seed),
        "Dirichlet a=0.5": partition_dirichlet(ytr, 5, 0.5, seed),
        "Dirichlet a=0.1": partition_dirichlet(ytr, 5, 0.1, seed),
        "2 users/client": partition_disjoint(ytr, k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--local-steps", type=int, nargs="+", default=[1, 5])
    args = ap.parse_args()
    act = Activation("exact")
    rows = {}
    curve_acc, curve_loss = {}, {}
    for seed in args.seeds:
        s = make_split(10, seed)
        k = s["k"]
        cen = evaluate(train_centralized(s["xtr"], s["ytr"], k, act=act), s["xte"], s["yte"], k)
        rows.setdefault(("centralized", "-", "-"), []).append(cen)
        for pname, clients in partitions(s["ytr"], k, seed).items():
            for steps in args.local_steps:
                rules = ["fedavg"] if steps == 1 else ["fedavg", "fedprox", "scaffold"]
                for rule in rules:
                    hist = []
                    w = train_fedavg(s["xtr"], s["ytr"], k, clients, act=act, local_steps=steps, rule=rule,
                                     history=hist, xte=s["xte"], yte=s["yte"])
                    rows.setdefault((pname, steps, rule), []).append(evaluate(w, s["xte"], s["yte"], k))
                    if seed == args.seeds[0] and rule == "fedavg" and steps == max(args.local_steps):
                        curve_acc[pname] = [100 * h["accuracy"] for h in hist]
                        curve_loss[pname] = [h["bce_loss"] for h in hist]
        print("seed", seed, "done", flush=True)
    table = []
    for (pname, steps, rule), runs in rows.items():
        row = {"partition": pname, "local_steps": steps, "rule": rule}
        for m in ("accuracy", "min_recall", "max_recall", "std_recall"):
            row[m] = mean_std([r[m] for r in runs])
        table.append(row)
        print(pname, steps, rule, {m: "%.2f+-%.2f" % (100 * row[m][0], 100 * row[m][1]) for m in ("accuracy", "min_recall", "std_recall")})
    save_json("heterogeneity_S%s.json" % "-".join(map(str, args.local_steps)), {"table": table, "curves_accuracy": curve_acc, "curves_loss": curve_loss})
    curves(curve_acc, "Test accuracy (%)", os.path.join(RESULTS, "fig_noniid_accuracy_S%d.pdf" % max(args.local_steps)))
    curves(curve_loss, "Cross-entropy loss", os.path.join(RESULTS, "fig_noniid_loss_S%d.pdf" % max(args.local_steps)))


if __name__ == "__main__":
    main()
