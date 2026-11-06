import json
import os

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(ms, digits=2):
    return f"{100 * ms[0]:.{digits}f} $\\pm$ {100 * ms[1]:.{digits}f}"


def pct_dict(d, digits=2):
    return f"{100 * d['mean']:.{digits}f} $\\pm$ {100 * d['std']:.{digits}f}"


def main():
    out = []
    m = load("main_results.json")
    if m:
        out.append("% Main results (5 splits)")
        names = {"centralized": "Centralized", "fedavg": "FL (FedAvg)", "fedavg_he": "FL (FedAvg) + HE"}
        for k, label in names.items():
            s = m["summary"][k]
            out.append(f"{label} & {pct_dict(s['accuracy'])} & {pct_dict(s['macro_recall'])} & {pct_dict(s['macro_precision'])} & {pct_dict(s['min_recall'])} \\\\")
    for tgt in ("strict", "relaxed"):
        v = load(f"verification_{tgt}.json")
        if not v:
            continue
        out.append(f"% Verification ({tgt})")
        for k, label in (("centralized", "Centralized"), ("fedavg", "FedAvg"), ("fedavg_he", "FedAvg + CKKS (linear sigmoid)")):
            s = v["summary"][k]
            out.append(f"{label} & {pct(s['eer_pooled_all'])} & {pct(s['eer_per_subject_mean'])} & {pct(s['eer_pooled_known'])} & {pct(s['eer_pooled_unseen'])} & {pct(s['frr_at_far_1pct'])} & {pct(s['frr_at_far_0.1pct'])} \\\\")
        s = v["summary"]["fedavg_he"]
        out.append("% risk levels FedAvg+HE: genuine / impostor")
        for lv in ("low", "medium", "high", "critical"):
            out.append(f"{lv} & {pct(s['genuine_' + lv], 1)} & {pct(s['impostor_' + lv], 1)} \\\\")
        out.append("% thresholds (FedAvg+HE): " + ", ".join(f"{t}={s['threshold_' + t][0]:.3f}+-{s['threshold_' + t][1]:.3f}" for t in ("theta1", "theta2", "theta3")))
        for t in ("theta1", "theta2", "theta3"):
            out.append(f"% {t}: far_known {pct(s[t + '_far_known'])}, far_unseen {pct(s[t + '_far_unseen'])}, frr {pct(s[t + '_frr'])}")
    for name in ("heterogeneity_S1-5.json", "heterogeneity_S20.json"):
        h = load(name)
        if h:
            out.append(f"% {name}")
            for r in h["table"]:
                out.append(f"{r['partition']} & {r['local_steps']} & {r['rule']} & {pct(r['accuracy'])} & {pct(r['min_recall'])} & {pct(r['std_recall'])} \\\\")
    s = load("sigmoid_ablation.json")
    if s:
        out.append("% sigmoid ablation")
        for r in s["table"]:
            out.append(f"{r['variant']} & {pct(r['accuracy'])} & {r['extra_depth_for_activation']} & {r['diverged_at_round']} & {r['logit_range_p0.5_p99.5']} \\\\")
    for rounds in (50, 200):
        sc = load(f"scaling_users_R{rounds}.json")
        if sc:
            out.append(f"% scaling users R={rounds}")
            for r in sc:
                out.append(f"{r['users']} & {r['samples']} & {pct(r['centralized'])} & {pct(r['fedavg'])} & {pct(r['fedavg_he'])} \\\\")
    p = load("poisoning.json")
    if p:
        out.append("% poisoning")
        for r in p:
            out.append(f"{r['scenario']} & {r['rate']} & {r['defense']} & {pct(r['accuracy'])} & {r['accuracy_drop_pp']:.2f} & {pct(r['asr'], 1)} & {pct(r['victim_far'], 1)} \\\\")
    inv = load("inversion.json")
    if inv:
        out.append("% inversion")
        for r in inv:
            out.append(f"{r['batch']} & {r['activation']} & {r['attack']} & {r['plaintext_mse'][0]:.3g} & {r['plaintext_corr'][0]:.3f} & {r['ciphertext_only_mse'][0]:.3g} & {r['ciphertext_only_corr'][0]:.3f} \\\\")
    for clip in ("1", "5"):
        d = load(f"dp_fedavg_clip{clip}.json")
        if d:
            out.append(f"% DP-FedAvg clip {clip}")
            for r in d:
                out.append(f"eps={r['epsilon']} sigma={r['sigma']:.2f} & {pct(r['accuracy'])} \\\\")
    for name in ("mlp.json", "incentive.json", "election_committee.json", "election_leader.json", "committee_weighted.json",
                 "ckks_bench_all.json", "ckks_fedavg_u10_s0_disjoint.json", "gas_anvil.json"):
        d = load(name)
        if d is not None:
            out.append(f"% {name}")
            text = json.dumps(d)
            out.append("% " + (text[:3000] + ("..." if len(text) > 3000 else "")))
    path = os.path.join(RES, "tables.tex")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
