import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACTS = os.path.join(ROOT, "contracts")
RESULTS = os.path.join(ROOT, "results")
RPC = "http://127.0.0.1:8545"
DEPLOYER_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPLOYER = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"


def rpc(method, params):
    req = urllib.request.Request(RPC, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["result"]


def cast(*args):
    return subprocess.run(["cast", *args], check=True, capture_output=True, text=True).stdout.strip()


def calldata(sig, *args):
    return cast("calldata", sig, *[str(a) for a in args])


def call(to, sig, *args):
    return cast("call", to, sig, *[str(a) for a in args], "--rpc-url", RPC)


def send(sender, to, sig, *args, value=0):
    tx = {"from": sender, "to": to, "data": calldata(sig, *args), "gas": hex(29_000_000)}
    if value:
        tx["value"] = hex(value)
    h = rpc("eth_sendTransaction", [tx])
    rc = rpc("eth_getTransactionReceipt", [h])
    if rc["status"] != "0x1":
        raise RuntimeError(f"revert {sig}")
    return int(rc["gasUsed"], 16)


def impersonate(addr):
    rpc("anvil_impersonateAccount", [addr])
    rpc("anvil_setBalance", [addr, hex(10 ** 21)])


def keyhex(key):
    return "0x" + format(key, "064x")


def addr_of(key):
    return cast("wallet", "address", "--private-key", keyhex(key))


def keccak(text):
    return cast("keccak", text)


class Bench:
    def __init__(self, k, max_batch, proof_required):
        self.k = k
        self.n = 3 * k
        env = dict(os.environ, COMMITTEE_SIZE=str(k), PROOF_REQUIRED="true" if proof_required else "false",
                   MAX_BATCH=str(max_batch), DEPLOYER_KEY=DEPLOYER_KEY, DEPLOY_OUT="deployments/latest.json",
                   ELECTION_WINDOW=str(self.n + 5))
        subprocess.run(["forge", "script", "script/Deploy.s.sol:Deploy", "--rpc-url", RPC, "--broadcast", "--offline",
                        "--slow", "-q"], cwd=CONTRACTS, env=env, check=True, capture_output=True)
        with open(os.path.join(CONTRACTS, "deployments", "latest.json")) as f:
            self.a = json.load(f)
        self.gas = {}
        self.val_keys = [0xB000 + i for i in range(self.n)]
        self.val_addrs = [addr_of(0xA000 + i) for i in range(self.n)]

    def record(self, name, g):
        self.gas.setdefault(name, []).append(g)

    def register_validators(self):
        for i in range(self.n):
            did = keccak(f"validator-{i}")
            self.record("SCid.register", send(DEPLOYER, self.a["id"], "register(bytes32,address,bytes32,bytes32)", did,
                                             self.val_addrs[i], keccak("vc"), keccak("cert")))
            pk = call(self.a["signer"], "publicKey(uint256)(bytes)", self.val_keys[i])
            impersonate(self.val_addrs[i])
            self.record("SCcommittee.register", send(self.val_addrs[i], self.a["committee"], "register(bytes,bytes32)", pk, did,
                                                    value=10 ** 18))

    def election(self, checkpoint="0x" + "00" * 32):
        self.record("SCcommittee.startEpoch", send(DEPLOYER, self.a["committee"], "startEpoch(bytes32)", checkpoint))
        seed = call(self.a["committee"], "seed()(bytes32)")
        for i in range(self.n):
            sigma = call(self.a["signer"], "sign(uint256,bytes32)(bytes)", self.val_keys[i], seed)
            self.record("SCcommittee.submitVrf", send(self.val_addrs[i], self.a["committee"], "submitVrf(bytes)", sigma))
        rpc("anvil_mine", [hex(self.n + 10)])
        self.record("SCcommittee.finalizeElection", send(DEPLOYER, self.a["committee"], "finalizeElection()"))
        self.epoch = int(call(self.a["committee"], "epoch()(uint64)").split()[0])
        self.leader = call(self.a["committee"], "leaderOf(uint64)(address)", self.epoch)
        members = call(self.a["committee"], "committee(uint64)(address[])", self.epoch)
        self.members = [m.strip() for m in members.strip("[]").split(",")]

    def quorum(self, message):
        q = 2 * ((len(self.members) - 1) // 3) + 1
        sks = [self.val_keys[self.val_addrs.index(m)] for m in self.members[:q]]
        sig = call(self.a["signer"], "aggregateSign(uint256[],bytes32)(bytes)", "[" + ",".join(map(str, sks)) + "]", message)
        return sig, (1 << q) - 1

    def requests(self, m, tag):
        batch = int(call(self.a["authreq"], "currentBatch()(uint256)").split()[0])
        for i in range(m):
            key = 0xD00000 + tag * 10000 + i
            dev = addr_of(key)
            did = keccak(f"device-{tag}-{i}")
            self.record("SCid.register", send(DEPLOYER, self.a["id"], "register(bytes32,address,bytes32,bytes32)", did, dev,
                                             keccak("vc"), keccak("cert")))
            ct = keccak(f"ciphertext-{tag}-{i}")
            nonce = keccak(f"nonce-{tag}-{i}")
            ts = int(rpc("eth_getBlockByNumber", ["latest", False])["timestamp"], 16)
            digest = call(self.a["authreq"], "requestDigest(bytes32,bytes32,bytes32,uint64)(bytes32)", did, ct, nonce, ts)
            sig = cast("wallet", "sign", "--private-key", keyhex(key), digest)
            r, s, v = "0x" + sig[2:66], "0x" + sig[66:130], int(sig[130:132], 16)
            self.record(f"SCauthreq.submitRequest", send(DEPLOYER, self.a["authreq"],
                                                        "submitRequest(bytes32,bytes32,bytes32,uint64,uint8,bytes32,bytes32)",
                                                        did, ct, nonce, ts, v, r, s))
        return batch

    def batch_round(self, m, tag):
        batch = self.requests(m, tag)
        root = keccak(f"root-{batch}")
        impersonate(self.leader)
        proof = "([1,2],[11559732032986387107991004021392285783925812861821192530917403151452391805634,10857046999023057135944570762232829481370756359578518086990519993285655852781,4082367875863433681332203403145435568316851327593401208105741076214120093531,8495653923123431417604973247489272438418190587263600148770280649306958101930],[1,2])"
        self.record(f"SCauthresp.submitEvaluation M={m}", send(self.leader, self.a["authresp"],
                                                                "submitEvaluation(uint256,bytes32,(uint256[2],uint256[4],uint256[2]))",
                                                                batch, root, proof))
        msg = call(self.a["verifier"], "batchMessage(bytes32,uint64,uint256)(bytes32)", root, self.epoch, batch)
        sig, bitmap = self.quorum(msg)
        self.record(f"SCverify.finalize K={self.k}", send(DEPLOYER, self.a["verifier"], "finalize(uint256,bytes,uint256)", batch, sig, bitmap))
        masked = "[" + ",".join(str((i * 7919 + 13) * 10 ** 12) for i in range(m)) + "]"
        dmsg = call(self.a["authresp"], "decryptionMessage(uint256,int256[])(bytes32)", batch, masked)
        dsig, dbit = self.quorum(dmsg)
        self.record(f"SCauthresp.postResults M={m}", send(DEPLOYER, self.a["authresp"], "postResults(uint256,int256[],bytes,uint256)",
                                                          batch, masked, dsig, dbit))
        return batch

    def checkpoint(self, batches):
        ids = "[" + ",".join(map(str, batches)) + "]"
        h = call(self.a["checkpoint"], "checkpointHash(uint64,uint256[])(bytes32)", self.epoch, ids)
        sig, bitmap = self.quorum(h)
        self.record(f"SCcheckpoint.anchor ({len(batches)} batches)", send(DEPLOYER, self.a["checkpoint"],
                                                                          "anchor(uint64,uint256[],bytes,uint256)", self.epoch, ids, sig, bitmap))

    def scores(self):
        vals = "[" + ",".join(self.members) + "]"
        sc = "[" + ",".join(str(70 + (i % 30)) for i in range(len(self.members))) + "]"
        msg = call(self.a["score"], "scoresMessage(uint64,address[],uint16[])(bytes32)", self.epoch, vals, sc)
        sig, bitmap = self.quorum(msg)
        self.record(f"SCscore.commitScores K={self.k}", send(DEPLOYER, self.a["score"], "commitScores(uint64,address[],uint16[],bytes,uint256)",
                                                             self.epoch, vals, sc, sig, bitmap))

    def double_sign(self):
        v = self.members[-1]
        sk = self.val_keys[self.val_addrs.index(v)]
        ra, rb = keccak("rootA"), keccak("rootB")
        ma = call(self.a["verifier"], "batchMessage(bytes32,uint64,uint256)(bytes32)", ra, self.epoch, 999)
        mb = call(self.a["verifier"], "batchMessage(bytes32,uint64,uint256)(bytes32)", rb, self.epoch, 999)
        sa = call(self.a["signer"], "sign(uint256,bytes32)(bytes)", sk, ma)
        sb = call(self.a["signer"], "sign(uint256,bytes32)(bytes)", sk, mb)
        self.record("SCslash.reportDoubleSign", send(DEPLOYER, self.a["slash"], "reportDoubleSign(address,uint64,uint256,bytes32,bytes,bytes32,bytes)",
                                                     v, self.epoch, 999, ra, sa, rb, sb))

    def revoke(self):
        did = keccak("revoke-me")
        self.record("SCid.register", send(DEPLOYER, self.a["id"], "register(bytes32,address,bytes32,bytes32)", did, DEPLOYER,
                                         keccak("vc"), keccak("cert")))
        self.record("SCid.revoke", send(DEPLOYER, self.a["id"], "revoke(bytes32)", did))


def start_anvil():
    proc = subprocess.Popen(["anvil", "--hardfork", "prague", "--silent", "--gas-limit", "60000000"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for _ in range(100):
        try:
            rpc("eth_chainId", [])
            return proc
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("anvil")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committees", type=int, nargs="+", default=[4, 7, 13])
    ap.add_argument("--batches", type=int, nargs="+", default=[10, 50, 100])
    args = ap.parse_args()
    summary = {}
    for k in args.committees:
        for m in args.batches:
            proc = start_anvil()
            try:
                b = Bench(k, m, False)
                b.register_validators()
                b.election()
                batches = [b.batch_round(m, t) for t in range(2)]
                b.scores()
                b.checkpoint(batches)
                b.double_sign()
                b.revoke()
                summary[f"K={k},M={m}"] = {name: sum(v) / len(v) for name, v in b.gas.items()}
                print(k, m, json.dumps(summary[f"K={k},M={m}"]), flush=True)
            finally:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(0.5)
    proc = start_anvil()
    try:
        b = Bench(4, 10, True)
        b.register_validators()
        b.election()
        batch = b.requests(10, 7)
        impersonate(b.leader)
        proof = "([1,2],[11559732032986387107991004021392285783925812861821192530917403151452391805634,10857046999023057135944570762232829481370756359578518086990519993285655852781,4082367875863433681332203403145435568316851327593401208105741076214120093531,8495653923123431417604973247489272438418190587263600148770280649306958101930],[1,2])"
        g = send(b.leader, b.a["authresp"], "submitEvaluation(uint256,bytes32,(uint256[2],uint256[4],uint256[2]))", batch, keccak("forged"), proof)
        summary["proof mode: submitEvaluation with invalid Groth16 proof (verify + slash)"] = g
        print("invalid proof", g)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "gas_anvil.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
