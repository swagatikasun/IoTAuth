import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def unilateral_ok(gain, p, stake_penalty, v, rep_penalty, ban, reward):
    return gain < p * (stake_penalty + v * rep_penalty + ban * reward)


def min_stake_penalty(g_max, p, f, v, rep_penalty, ban, reward):
    return max(0.0, g_max / (p * (f + 1)) - v * rep_penalty - ban * reward)


def main():
    base = {"reward_per_epoch": 1.0, "value_per_reputation_point": 0.5, "reputation_penalty": 20.0,
            "ban_epochs": 100, "detection_probability": 1.0}
    rows = []
    for k in (4, 7, 13, 31, 61, 103):
        f = (k - 1) // 3
        for g_max in (1_000.0, 10_000.0, 100_000.0):
            lam = min_stake_penalty(g_max, base["detection_probability"], f, base["value_per_reputation_point"],
                                    base["reputation_penalty"], base["ban_epochs"], base["reward_per_epoch"])
            rows.append({"K": k, "f": f, "G_max": g_max, "min_stake_penalty": lam,
                         "per_validator_loss_at_min": lam + base["value_per_reputation_point"] * base["reputation_penalty"]
                         + base["ban_epochs"] * base["reward_per_epoch"]})
    example = {"K": 31, "f": 10, "G_max": 10_000.0}
    example["min_stake_penalty"] = min_stake_penalty(10_000.0, 1.0, 10, 0.5, 20.0, 100, 1.0)
    example["unilateral_bribe_deterred_up_to"] = 1.0 * (example["min_stake_penalty"] + 0.5 * 20.0 + 100 * 1.0)
    out = {"assumptions": base, "table": rows, "worked_example": example}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "incentive.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows:
        print(r)
    print(example)


if __name__ == "__main__":
    main()
