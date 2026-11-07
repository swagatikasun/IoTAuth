import json
import os
import time
import numpy as np
from committee import RESULTS, hyper_tail, weighted_tail, five_level_population, worst_case_split, random_split_expectation


def main():
    rows = []
    n = 100
    for k in (13, 31, 61):
        for ratio in (0.20, 0.30):
            t = int(round(ratio * n))
            rho = five_level_population(n, k)
            byz, hon = worst_case_split(rho, t)
            start = time.time()
            mean, std = random_split_expectation(rho, t, k, samples=50)
            rows.append({"n": n, "K": k, "t_over_n": ratio, "t": t,
                         "uniform": hyper_tail(n, t, k),
                         "weighted_random_placement": mean,
                         "weighted_random_placement_std": std,
                         "weighted_worst_case": weighted_tail(byz, hon, k),
                         "seconds": time.time() - start})
            print(json.dumps(rows[-1]))
    with open(os.path.join(RESULTS, "committee_grid.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
