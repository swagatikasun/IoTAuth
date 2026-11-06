import argparse
import json
import os
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

REST = {"leader": 0.05, "committee": 0.30, 1: 1.00, 2: 1.20, 3: 1.50}


def rest_multiplier(last_role, rested):
    if last_role == "leader":
        return REST["leader"]
    if last_role == "committee":
        return REST["committee"]
    return REST[min(max(rested, 1), 3)]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


class Election:
    def __init__(self, n, k, t, committee_rule, leader_rule, adversary="static", seed=0,
                 w_perf=0.6, w_cred=0.3, w_fair=0.1, cap_window=None, cap_leads=None,
                 credit=10.0, delta=0.86, q_low=60.0, q_high=95.0, noise=2.0, power=1.0):
        self.n, self.k, self.t = n, k, t
        self.f = (k - 1) // 3
        self.rng = np.random.default_rng(seed)
        self.committee_rule = committee_rule
        self.leader_rule = leader_rule
        self.adversary = adversary
        self.w = (w_perf, w_cred, w_fair)
        self.cap_window = cap_window
        self.cap_leads = cap_leads
        self.credit = credit
        self.delta = delta
        self.noise = noise
        self.power = power
        self.quality = self.rng.uniform(q_low, q_high, n)
        self.byz = np.zeros(n, dtype=bool)
        self.byz[self.rng.choice(n, t, replace=False)] = True
        self.last_role = np.array(["rest"] * n, dtype=object)
        self.rested = np.full(n, 3)
        self.scores = [[] for _ in range(n)]
        self.credits = np.zeros(n)
        self.last_led = np.full(n, -10 ** 9)
        self.led_ever = np.zeros(n, dtype=bool)
        self.lead_log = []
        self.static_committee = None
        self.prev_leader = None

    def multipliers(self):
        return np.array([rest_multiplier(self.last_role[j], self.rested[j]) for j in range(self.n)])

    def form_committee(self):
        if self.committee_rule == "static":
            if self.static_committee is None:
                self.static_committee = self.rng.choice(self.n, self.k, replace=False)
            return self.static_committee
        if self.committee_rule == "merit":
            perf = self.quality + self.rng.normal(0, 0.5, self.n)
            return np.argsort(-perf)[: self.k]
        y = self.rng.uniform(0, 1, self.n)
        if self.committee_rule == "uniform":
            return np.argsort(-y)[: self.k]
        return np.argsort(-(y * self.multipliers()))[: self.k]

    def place_adversary(self):
        if self.adversary != "mobile":
            return
        if self.committee_rule == "weighted":
            rho = self.multipliers()
            order = np.lexsort((self.rng.uniform(size=self.n), -rho))
        elif self.committee_rule in ("static", "merit"):
            order = np.argsort(-self.quality) if self.committee_rule == "merit" else None
            if self.committee_rule == "static" and self.static_committee is not None:
                rest = np.setdiff1d(np.arange(self.n), self.static_committee)
                order = np.concatenate([self.static_committee, rest])
            if order is None:
                order = self.rng.permutation(self.n)
        else:
            order = self.rng.permutation(self.n)
        self.byz[:] = False
        self.byz[order[: self.t]] = True

    def perf(self, j):
        s = self.scores[j][-3:]
        return float(np.mean(s)) if s else 0.0

    def fair(self, j, epoch):
        if not self.led_ever[j]:
            return 10.0
        if epoch - self.last_led[j] >= 5:
            return 5.0
        return 0.0

    def leader_score(self, j, epoch):
        a, b, c = self.w
        return a * self.perf(j) + b * self.credits[j] + c * self.fair(j, epoch)

    def capped(self, j, epoch):
        if self.cap_window is None:
            return False
        recent = [l for (e, l) in self.lead_log[-self.cap_window:] if l == j and epoch - e <= self.cap_window]
        return len(recent) >= self.cap_leads

    def choose_leader(self, committee, epoch):
        cand = [j for j in committee if j != self.prev_leader]
        capped = [j for j in cand if not self.capped(j, epoch)]
        if capped:
            cand = capped
        if self.leader_rule == "random":
            return int(self.rng.choice(cand))
        scores = np.array([self.leader_score(j, epoch) for j in cand])
        if self.leader_rule == "merit":
            return int(cand[int(np.argmax(scores))])
        p = np.maximum(scores, 1e-9) ** self.power
        p = p / p.sum()
        return int(self.rng.choice(cand, p=p))

    def step(self, epoch):
        self.place_adversary()
        committee = np.asarray(self.form_committee())
        x = int(self.byz[committee].sum())
        leader = self.choose_leader(committee, epoch)
        for j in committee:
            self.scores[j].append(float(np.clip(self.quality[j] + self.rng.normal(0, self.noise), 0, 100)))
        self.credits *= self.delta
        if self.scores[leader][-1] >= 70:
            self.credits[leader] += self.credit
        self.led_ever[leader] = True
        self.last_led[leader] = epoch
        self.lead_log.append((epoch, leader))
        in_c = np.zeros(self.n, dtype=bool)
        in_c[committee] = True
        for j in range(self.n):
            if j == leader:
                self.last_role[j] = "leader"
                self.rested[j] = 0
            elif in_c[j]:
                self.last_role[j] = "committee"
                self.rested[j] = 0
            else:
                self.last_role[j] = "rest"
                self.rested[j] += 1
        self.prev_leader = leader
        return set(committee.tolist()), x, leader

    def run(self, epochs):
        fails = 0
        overlaps = []
        prev = None
        counts = np.zeros(self.n)
        lq = []
        for e in range(epochs):
            c, x, leader = self.step(e)
            fails += x > self.f
            if prev is not None:
                overlaps.append(len(c & prev) / self.k)
            prev = c
            counts[leader] += 1
            lq.append(self.quality[leader])
        top = np.sort(counts)[::-1]
        m = max(1, int(round(0.1 * self.n)))
        return {
            "fail": fails / epochs,
            "overlap": float(np.mean(overlaps)),
            "max_leader_share": float(counts.max() / epochs),
            "gini": gini(counts),
            "top10_share": float(top[:m].sum() / epochs),
            "mean_leader_quality": float(np.mean(lq)),
            "population_quality": float(self.quality.mean()),
            "distinct_leaders": int((counts > 0).sum()),
        }


def ablation_committee(epochs, seed):
    rows = []
    for n, t in [(100, 30), (200, 60)]:
        for name, rule in [("static committee", "static"), ("merit-only top-K", "merit"),
                           ("uniform VRF", "uniform"), ("VRF + rest (ours)", "weighted")]:
            res = {}
            for adv in ["static", "mobile"]:
                sim = Election(n, 31, t, rule, "random", adversary=adv, seed=seed)
                res[adv] = sim.run(epochs)
            rows.append({"n": n, "t": t, "rule": name, "fail_random": res["static"]["fail"],
                         "fail_mobile": res["mobile"]["fail"], "overlap": res["static"]["overlap"],
                         "max_leader_share": res["static"]["max_leader_share"]})
    return rows


def ablation_leader(epochs, seeds):
    variants = [
        ("A uniform committee + random leader", "uniform", "random", {}),
        ("B VRF+rest committee + random leader", "weighted", "random", {}),
        ("C VRF+rest committee + merit leader", "weighted", "merit", {}),
        ("D uniform committee + merit leader", "uniform", "merit", {}),
        ("E VRF+rest + merit, fairness 0.3 + cap", "weighted", "merit",
         {"w_perf": 0.5, "w_cred": 0.2, "w_fair": 0.3, "cap_window": 50, "cap_leads": 1}),
        ("F VRF+rest + leader drawn with prob. prop. to L_j", "weighted", "weighted", {}),
        ("G VRF+rest + prop. to L_j + cap", "weighted", "weighted", {"cap_window": 50, "cap_leads": 1}),
        ("H VRF+rest + prop. to L_j^10 + cap", "weighted", "weighted", {"cap_window": 50, "cap_leads": 1, "power": 10.0}),
    ]
    rows = []
    for name, cr, lr, kw in variants:
        runs = [Election(100, 31, 30, cr, lr, seed=s, **kw).run(epochs) for s in seeds]
        row = {"variant": name}
        for key in runs[0]:
            vals = [r[key] for r in runs]
            row[key] = float(np.mean(vals))
            row[key + "_std"] = float(np.std(vals))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["committee", "leader"], required=True)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if args.part == "committee":
        out = ablation_committee(args.epochs, 0)
    else:
        out = ablation_leader(args.epochs, list(range(args.seeds)))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"election_{args.part}.json"), "w") as f:
        json.dump(out, f, indent=2)
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("_std")})


if __name__ == "__main__":
    main()
