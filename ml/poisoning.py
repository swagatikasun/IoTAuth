import argparse
import numpy as np
from common import (Activation, add_bias, evaluate, feature_columns, mean_std, one_hot, partition_disjoint,
                    save_json, select_users, train_fedavg, Preprocessor)
from sklearn.model_selection import train_test_split
from verification import threshold_at_far

ATTACKER = 0
VICTIM = 9
TRIGGER = ["H.period", "H.t", "H.i"]
TRIGGER_SHIFT = 0.25


def build(seed):
    df, enrolled, unseen = select_users(10)
    cols = feature_columns(df)
    d = df[df["subject"].isin(enrolled)].reset_index(drop=True)
    y = d["subject"].map({s: i for i, s in enumerate(enrolled)}).to_numpy()
    raw = d[cols].to_numpy(dtype=float)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=seed)
    pre = Preprocessor().fit(raw[tr])
    u = df[df["subject"].isin(unseen)]
    rng = np.random.default_rng(seed)
    cal_subj = rng.permutation(np.unique(u["subject"]))[:20]
    cal = u[u["subject"].isin(cal_subj)][cols].to_numpy(dtype=float)
    trig = np.array([cols.index(c) for c in TRIGGER])
    return {"raw": raw, "y": y, "tr": tr, "va": va, "te": te, "pre": pre, "cal": pre.transform(cal), "trig": trig}


def triggered(raw, trig):
    out = raw.copy()
    out[:, trig] += TRIGGER_SHIFT
    return out


def poisoned_client(data, clients, scenario, rate, provenance, seed):
    rng = np.random.default_rng(seed + 7)
    idx = clients[ATTACKER]
    raw = data["raw"][data["tr"]][idx].copy()
    y = data["y"][data["tr"]][idx].copy()
    own = np.unique(y)
    target = own[-1] if provenance else VICTIM
    n_poison = int(round(rate * len(y)))
    pick = rng.choice(len(y), n_poison, replace=False)
    if scenario == "label_flip":
        if provenance:
            y[pick] = np.where(y[pick] == own[0], own[-1], own[0])
        else:
            y[pick] = VICTIM
    else:
        raw[pick] = triggered(raw[pick], data["trig"])
        y[pick] = target
    return data["pre"].transform(raw), y


def leave_one_out_gate(xva, yva, k, delta):
    xa = add_bias(xva)

    def acc(w):
        return float(np.mean(np.argmax(xa @ w, 1) == yva))

    def gate(w, v, deltas, sizes, keep):
        if len(keep) <= 1:
            return keep
        sizes_arr = np.array(sizes, dtype=float)

        def agg(ids):
            a = sizes_arr[ids] / sizes_arr[ids].sum()
            return sum(ai * deltas[i] for ai, i in zip(a, ids))

        full = acc(w - 0.9 * v - agg(keep))
        out = []
        for i in keep:
            rest = [j for j in keep if j != i]
            if acc(w - 0.9 * v - agg(rest)) - full <= delta:
                out.append(i)
        return out if out else keep
    return gate


def run_one(data, scenario, rate, defense, k, norm_bound, seed):
    ytr = data["y"][data["tr"]]
    xtr = data["pre"].transform(data["raw"][data["tr"]])
    clients = partition_disjoint(ytr, k)
    provenance = defense in ("provenance", "provenance+norm", "all")
    use_norm = defense in ("provenance+norm", "norm", "all")
    use_gate = defense in ("gate", "all")
    transform = None
    if scenario != "clean":
        px, py = poisoned_client(data, clients, scenario, rate, provenance, seed)
        t = one_hot(py, k)
        transform = {ATTACKER: lambda xi, ti, px=px, t=t: (add_bias(px), t)}
    gate = leave_one_out_gate(data["pre"].transform(data["raw"][data["va"]]), data["y"][data["va"]], k, 0.005) if use_gate else None
    w = train_fedavg(xtr, ytr, k, clients, act=Activation("exact"), transform=transform,
                     norm_bound=norm_bound if use_norm else None, gate=gate)
    xte = data["pre"].transform(data["raw"][data["te"]])
    yte = data["y"][data["te"]]
    acc = evaluate(w, xte, yte, k)["accuracy"]
    theta2 = threshold_at_far((add_bias(data["cal"]) @ w).ravel(), 0.01)
    att = np.isin(yte, [0, 1])
    if scenario == "backdoor":
        probe = data["pre"].transform(triggered(data["raw"][data["te"]][att], data["trig"]))
    else:
        probe = xte[att]
    s = add_bias(probe) @ w
    asr = float(np.mean(np.argmax(s, 1) == VICTIM))
    victim_far = float(np.mean(s[:, VICTIM] >= theta2))
    return {"accuracy": acc, "asr": asr, "victim_far": victim_far}


def clean_norm_bound(data, k):
    ytr = data["y"][data["tr"]]
    xtr = data["pre"].transform(data["raw"][data["tr"]])
    norms = []
    train_fedavg(xtr, ytr, k, partition_disjoint(ytr, k), act=Activation("exact"), norms=norms)
    return 1.1 * float(np.max(norms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    k = 10
    scenarios = [("clean", 0.0)] + [("label_flip", r) for r in (0.1, 0.2, 0.4)] + [("backdoor", r) for r in (0.1, 0.2, 0.4)]
    defenses = ["none", "provenance", "norm", "gate", "all"]
    res = {}
    for seed in args.seeds:
        data = build(seed)
        nb = clean_norm_bound(data, k)
        for sc, rate in scenarios:
            for df in (["none"] if sc == "clean" else defenses):
                r = run_one(data, sc, rate, df, k, nb, seed)
                res.setdefault((sc, rate, df), []).append(r)
        print("seed", seed, "done", flush=True)
    base = mean_std([r["accuracy"] for r in res[("clean", 0.0, "none")]])[0]
    table = []
    for (sc, rate, df), runs in res.items():
        row = {"scenario": sc, "rate": rate, "defense": df,
               "accuracy": mean_std([r["accuracy"] for r in runs]),
               "accuracy_drop_pp": 100 * (base - mean_std([r["accuracy"] for r in runs])[0]),
               "asr": mean_std([r["asr"] for r in runs]),
               "victim_far": mean_std([r["victim_far"] for r in runs])}
        table.append(row)
        print(sc, rate, df, "acc %.2f drop %.2f asr %.2f vfar %.2f" % (100 * row["accuracy"][0], row["accuracy_drop_pp"], 100 * row["asr"][0], 100 * row["victim_far"][0]))
    save_json("poisoning.json", table)


if __name__ == "__main__":
    main()
