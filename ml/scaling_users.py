import argparse
from common import (Activation, evaluate, make_split, mean_std, partition_groups, save_json,
                    train_centralized, train_fedavg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, nargs="+", default=[10, 20, 30, 51])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()
    table = []
    for n in args.users:
        acc = {"centralized": [], "fedavg": [], "fedavg_he": []}
        for seed in args.seeds:
            s = make_split(n, seed)
            k = s["k"]
            clients = partition_groups(s["ytr"], k)
            acc["centralized"].append(evaluate(train_centralized(s["xtr"], s["ytr"], k, rounds=args.rounds, act=Activation("exact")), s["xte"], s["yte"], k)["accuracy"])
            acc["fedavg"].append(evaluate(train_fedavg(s["xtr"], s["ytr"], k, clients, rounds=args.rounds, act=Activation("exact")), s["xte"], s["yte"], k)["accuracy"])
            acc["fedavg_he"].append(evaluate(train_fedavg(s["xtr"], s["ytr"], k, clients, rounds=args.rounds, act=Activation("linear")), s["xte"], s["yte"], k)["accuracy"])
        row = {"users": n, "samples": 400 * n}
        row.update({m: mean_std(v) for m, v in acc.items()})
        table.append(row)
        print(n, {m: "%.2f+-%.2f" % (100 * row[m][0], 100 * row[m][1]) for m in acc})
    save_json("scaling_users_R%d.json" % args.rounds, table)


if __name__ == "__main__":
    main()
