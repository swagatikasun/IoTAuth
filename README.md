# IoTAuth artifact

Code, contracts and scripts used for the evaluation of "IoTAuth: A Secure and Scalable Framework for Decentralized IoT Authentication".

## Layout

| Path | Content |
|---|---|
| `data/` | Place the CMU Keystroke Dynamics Benchmark file `DSL-StrongPasswordData.csv` here (51 subjects x 400 repetitions; see `data/README.md`) |
| `ml/common.py` | Loading, preprocessing (standardise, PCA-20, degree-2 interaction terms, standardise), splits, one-vs-rest logistic regression, FedAvg with server momentum, FedProx, SCAFFOLD |
| `ml/ckks_fedavg.py` | Encrypted federated training with OpenFHE CKKS (features, labels and per-sample products stay encrypted; only the aggregated gradient is decrypted each round) |
| `ml/run_main.py` | Centralized, FedAvg and FedAvg+HE over five stratified splits; Fig. 3 and Fig. 4 |
| `ml/verification.py` | FAR, FRR, EER with known and unseen impostors, risk-threshold calibration, risk-level distribution |
| `ml/heterogeneity.py` | IID, Dirichlet and 2-users-per-client partitions; 1, 5 and 20 local steps; FedProx and SCAFFOLD |
| `ml/sigmoid_ablation.py` | Exact sigmoid, linear approximation, degree-3 least-squares and minimax approximations |
| `ml/scaling_users.py` | 10, 20, 30 and 51 enrolled users |
| `ml/mlp.py` | Square-activation MLP: plaintext federated training and CKKS inference |
| `ml/poisoning.py` | Label flipping and backdoor attacks with provenance, norm clipping and accuracy-gate defences |
| `ml/inversion.py` | Gradient inversion on plaintext updates versus a ciphertext-only adversary |
| `ml/dp_fedavg.py` | DP-FedAvg baseline (per-sample clipping, Gaussian noise, RDP accounting) |
| `ml/ckks_bench.py` | Device encryption, per-request evaluation, mask headroom, threshold decryption, wrong-key decryption |
| `ml/deployed_timing.py` | Device encryption time and request size with the deployed parameter set (needs OpenFHE built with `NATIVE_SIZE=128`) |
| `ml/decision_params.py` | Masked threshold decryption with noise flooding (OpenFHE built with 128-bit native integers) |
| `ml/make_tables.py` | Collects all result files into `results/tables.tex` |
| `election/committee.py` | Exact committee failure probability (uniform and weighted election) |
| `election/committee_grid.py` | Weighted rule at K = 13, 31, 61 and t/n = 0.2, 0.3 |
| `election/simulate.py` | Election ablation: committee rules and leader rules, including the fairness corrections |
| `election/incentive.py` | Numeric example for the slashing analysis |
| `contracts/` | The nine contracts (Foundry), BLS12-381 verification through EIP-2537 precompiles, Groth16 verifier, tests |
| `testbed/gas_bench.py` | Per-transaction gas on a local anvil chain (Prague hard fork) |
| `testbed/bls_bench.c` | Off-chain BLS signing, aggregation and verification time (blst) |
| `proverif/iotauth.pv` | ProVerif model of request submission, masked decryption and result delivery |
| `caliper/` | Caliper workload for the new `submitRequest` signature |
| `results/` | Result files (JSON and logs) produced by the commands below |

## Requirements

- Python 3.10 or newer with numpy, scipy, pandas, scikit-learn, matplotlib.
- OpenFHE 1.3.1 and openfhe-python 1.3.1.0 for every file that imports `openfhe`.
- Foundry 1.5 or newer (`forge`, `anvil`, `cast`) for the contracts and the gas benchmark.
- ProVerif 2.05 (`opam install proverif`) for the formal model.
- blst (https://github.com/supranational/blst) for `testbed/bls_bench.c`.

## Running

Each block below is one command and takes at most about an hour on a laptop; check the printed summary before starting the next block. Outputs go to `results/`.

Encrypted training for split 0 (ten blocks of five rounds; `--resume` continues from the checkpoint written after every round):

```
cd ml && python3 ckks_fedavg.py --seed 0 --rounds 50 --batch-rounds 5 --resume
```

Per-sidechain cost with client-side summation and packing (one packed ciphertext uploaded per sidechain and round):

```
cd ml && python3 ckks_fedavg.py --seed 0 --rounds 2 --batch-rounds 2 --reduce client
```

The same two measurements with the deployed parameter set (OpenFHE built with `-DNATIVE_SIZE=128`):

```
cd ml && python3 ckks_fedavg.py --seed 0 --rounds 2 --batch-rounds 2 --reduce client --tag-suffix _native128 && python3 deployed_timing.py
```

Plaintext experiments, block 1 (uses the encrypted weights of split 0 when they exist):

```
cd ml && python3 run_main.py && python3 verification.py --targets strict && python3 verification.py --targets relaxed && python3 heterogeneity.py --local-steps 1 5 && python3 heterogeneity.py --local-steps 20
```

Block 2:

```
cd ml && python3 sigmoid_ablation.py && python3 scaling_users.py --rounds 50 && python3 scaling_users.py --rounds 200 && python3 poisoning.py && python3 dp_fedavg.py --clip 1
```

Block 3:

```
cd ml && python3 dp_fedavg.py --clip 5 && python3 inversion.py --trials 5 && python3 mlp.py && python3 ckks_bench.py --part all && python3 make_tables.py
```

Block 4 (election and incentives):

```
cd election && python3 committee.py --part all && python3 committee_grid.py && python3 simulate.py --part committee && python3 simulate.py --part leader && python3 incentive.py
```

Block 5 (contracts; needs Foundry):

```
cd contracts && forge test -vv && cd .. && python3 testbed/gas_bench.py --committees 4 7 13 31 --batches 10 100
```

Decision-path parameters (needs OpenFHE built with `-DNATIVE_SIZE=128` and its Python bindings; point `LD_LIBRARY_PATH` and `PYTHONPATH` to that build):

```
cd ml && LD_LIBRARY_PATH=/opt/openfhe128/lib PYTHONPATH=/opt/py128 python3 decision_params.py
```

Off-chain BLS timing and the ProVerif model:

```
git clone --depth 1 https://github.com/supranational/blst.git /tmp/blst && (cd /tmp/blst && ./build.sh) && gcc -O2 -I/tmp/blst/bindings testbed/bls_bench.c /tmp/blst/libblst.a -o /tmp/bls_bench && /tmp/bls_bench && proverif proverif/iotauth.pv
```

Expected ProVerif result (full output in `results/proverif.log`): both secrecy queries, non-injective agreement and result authenticity are true; injective agreement is false because ProVerif tables are not atomic (two contract copies read the nonce table before either writes it). On the chain the check and the write are in one transaction; `testReplayAndClockRejected` in `contracts/test/Protocol.t.sol` checks that a replayed request reverts.
