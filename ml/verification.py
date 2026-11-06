import argparse
import os
import numpy as np
from common import (Activation, add_bias, make_split, mean_std, partition_disjoint, save_json,
                    train_centralized, train_fedavg, RESULTS)
from plots import roc

TARGETS = {"theta1": 0.001, "theta2": 0.01, "theta3": 0.10}
TARGET_SETS = {"strict": (0.001, 0.01, 0.10), "relaxed": (0.01, 0.05, 0.20)}


def claim_scores(w, x, claims):
    s = add_bias(x) @ w
    return s[np.arange(len(claims)), claims]


def impostor_matrix(w, x, k):
    s = add_bias(x) @ w
    return s.reshape(-1, 1).ravel() if k == 1 else s.ravel()


def threshold_at_far(impostor, far):
    srt = np.sort(impostor)[::-1]
    idx = int(np.floor(far * len(srt)))
    if idx >= len(srt):
        return -np.inf
    return float(srt[idx]) + 1e-12


def rates(genuine, impostor, thr):
    return float(np.mean(impostor >= thr)), float(np.mean(genuine < thr))


def curve(genuine, impostor, points=2000):
    allv = np.concatenate([genuine, impostor])
    thr = np.unique(np.quantile(allv, np.linspace(0, 1, points)))
    far = np.array([np.mean(impostor >= t) for t in thr])
    frr = np.array([np.mean(genuine < t) for t in thr])
    return thr, far, frr


def eer(genuine, impostor):
    thr, far, frr = curve(genuine, impostor)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def frr_at_far(genuine, impostor, far_target):
    return rates(genuine, impostor, threshold_at_far(impostor, far_target))[1]


def evaluate_model(w, s, calib_idx, test_idx):
    k = s["k"]
    yte = s["yte"]
    gen = claim_scores(w, s["xte"], yte)
    known = []
    known_claim = []
    sc = add_bias(s["xte"]) @ w
    for u in range(k):
        mask = yte != u
        known.append(sc[mask, u])
        known_claim.append(np.full(mask.sum(), u))
    known = np.concatenate(known)
    su = add_bias(s["xunseen"]) @ w
    unseen_cal = su[calib_idx].ravel()
    unseen_test = su[test_idx].ravel()
    thetas = {name: threshold_at_far(unseen_cal, far) for name, far in TARGETS.items()}
    impostor_all = np.concatenate([known, unseen_test])
    per_subject = []
    for u in range(k):
        g = gen[yte == u]
        imp = np.concatenate([sc[yte != u, u], su[test_idx, u]])
        per_subject.append(eer(g, imp))
    out = {
        "eer_pooled_all": eer(gen, impostor_all),
        "eer_pooled_known": eer(gen, known),
        "eer_pooled_unseen": eer(gen, unseen_test),
        "eer_per_subject_mean": float(np.mean(per_subject)),
        "eer_per_subject_std": float(np.std(per_subject)),
        "frr_at_far_1pct": frr_at_far(gen, impostor_all, 0.01),
        "frr_at_far_0.1pct": frr_at_far(gen, impostor_all, 0.001),
        "thresholds": thetas,
    }
    for name, thr in thetas.items():
        far_k, frr = rates(gen, known, thr)
        far_u, _ = rates(gen, unseen_test, thr)
        out[f"operational_{name}"] = {"target_far": TARGETS[name], "far_known": far_k, "far_unseen": far_u, "frr": frr}
    t1, t2, t3 = thetas["theta1"], thetas["theta2"], thetas["theta3"]

    def levels(v):
        return {
            "low": float(np.mean(v >= t1)),
            "medium": float(np.mean((v >= t2) & (v < t1))),
            "high": float(np.mean((v >= t3) & (v < t2))),
            "critical": float(np.mean(v < t3)),
        }

    out["levels_genuine"] = levels(gen)
    out["levels_impostor"] = levels(impostor_all)
    _, far_c, frr_c = curve(gen, impostor_all)
    return out, (far_c, frr_c)


def ckks_weights(seed):
    path = os.path.join(RESULTS, f"ckks_weights_u10_s{seed}_disjoint.npy")
    return np.load(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--targets", choices=list(TARGET_SETS), default="strict")
    args = ap.parse_args()
    TARGETS.update(dict(zip(("theta1", "theta2", "theta3"), TARGET_SETS[args.targets])))
    per_model = {"centralized": [], "fedavg": [], "fedavg_he": []}
    for seed in args.seeds:
        s = make_split(10, seed)
        k = s["k"]
        clients = partition_disjoint(s["ytr"], k)
        subjects = np.unique(s["unseen_subject"])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(subjects)
        cal_subj, test_subj = perm[:20], perm[20:]
        calib_idx = np.isin(s["unseen_subject"], cal_subj)
        test_idx = np.isin(s["unseen_subject"], test_subj)
        enc = ckks_weights(seed)
        models = {
            "centralized": train_centralized(s["xtr"], s["ytr"], k, act=Activation("exact")),
            "fedavg": train_fedavg(s["xtr"], s["ytr"], k, clients, act=Activation("exact")),
            "fedavg_he": enc if enc is not None else train_fedavg(s["xtr"], s["ytr"], k, clients, act=Activation("linear")),
        }
        curves = {}
        for name, w in models.items():
            res, c = evaluate_model(w, s, calib_idx, test_idx)
            res["seed"] = seed
            per_model[name].append(res)
            curves[name] = c
        if seed == args.seeds[0]:
            roc({"Centralized": curves["centralized"], "FedAvg": curves["fedavg"], "FedAvg + HE": curves["fedavg_he"]},
                os.path.join(RESULTS, "fig_roc.pdf"))
        print(seed, {n: round(per_model[n][-1]["eer_pooled_all"], 4) for n in per_model})
    summary = {}
    for name, runs in per_model.items():
        row = {}
        for key in ("eer_pooled_all", "eer_pooled_known", "eer_pooled_unseen", "eer_per_subject_mean",
                    "frr_at_far_1pct", "frr_at_far_0.1pct"):
            row[key] = mean_std([r[key] for r in runs])
        for lv in ("low", "medium", "high", "critical"):
            row["genuine_" + lv] = mean_std([r["levels_genuine"][lv] for r in runs])
            row["impostor_" + lv] = mean_std([r["levels_impostor"][lv] for r in runs])
        for th in TARGETS:
            row["threshold_" + th] = mean_std([r["thresholds"][th] for r in runs])
            for q in ("far_known", "far_unseen", "frr"):
                row[f"{th}_{q}"] = mean_std([r[f"operational_{th}"][q] for r in runs])
        summary[name] = row
    save_json(f"verification_{args.targets}.json", {"runs": per_model, "summary": summary})
    for name, row in summary.items():
        print(name, {k: "%.2f+-%.2f" % (100 * v[0], 100 * v[1]) for k, v in row.items() if not k.startswith("threshold")})
        print(name, {k: "%.3f+-%.3f" % v for k, v in row.items() if k.startswith("threshold")})


if __name__ == "__main__":
    main()
