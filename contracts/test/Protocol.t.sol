// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test, console2} from "forge-std/Test.sol";
import {BLS} from "../src/BLS.sol";
import {Groth16} from "../src/Groth16.sol";
import {SCid} from "../src/SCid.sol";
import {SCcommittee, IScore} from "../src/SCcommittee.sol";
import {SCverify, IAuthResp} from "../src/SCverify.sol";
import {SCauthreq} from "../src/SCauthreq.sol";
import {SCauthresp, ISlash, IModel} from "../src/SCauthresp.sol";
import {SCscore, ISlashTraining} from "../src/SCscore.sol";
import {SCslash} from "../src/SCslash.sol";
import {SCcheckpoint} from "../src/SCcheckpoint.sol";
import {SCmodel, ISlashAggregation} from "../src/SCmodel.sol";
import {BLSHelper} from "./BLSHelper.sol";

contract Groth16Harness {
    using Groth16 for Groth16.VerifyingKey;

    Groth16.VerifyingKey private vk;

    constructor(uint256 inputs) {
        vk.alpha = [uint256(1), 2];
        vk.beta = Groth16.g2Generator();
        vk.gamma = Groth16.g2Generator();
        vk.delta = Groth16.g2Generator();
        for (uint256 i = 0; i <= inputs; i++) vk.ic.push([uint256(1), 2]);
    }

    function verify(Groth16.Proof calldata p, uint256[] calldata inputs) external view returns (bool) {
        return vk.verify(p, inputs);
    }
}

contract ProtocolTest is Test {
    struct Deployment {
        SCid id;
        SCcommittee committee;
        SCverify verifier;
        SCauthreq authreq;
        SCauthresp authresp;
        SCscore score;
        SCslash slash;
        SCcheckpoint checkpoint;
        SCmodel model;
    }

    uint256 internal constant STAKE = 1 ether;

    function dummyProof() internal pure returns (Groth16.Proof memory p) {
        p.a = [uint256(1), 2];
        p.b = Groth16.g2Generator();
        p.c = [uint256(1), 2];
    }

    function vkIc(uint256 inputs) internal pure returns (uint256[2][] memory ic) {
        ic = new uint256[2][](inputs + 1);
        for (uint256 i = 0; i <= inputs; i++) ic[i] = [uint256(1), 2];
    }

    function deploy(uint256 k, bool proofRequired) internal returns (Deployment memory d) {
        d.id = new SCid(address(this), address(this), keccak256("ca-key"));
        d.committee = new SCcommittee(d.id, k, STAKE, 5, false, 5, address(this));
        d.verifier = new SCverify(d.committee, address(this));
        d.authreq = new SCauthreq(d.id, d.committee, 100, 2, 30);
        d.authresp = new SCauthresp(d.authreq, d.committee, d.verifier, proofRequired, address(this));
        d.score = new SCscore(d.committee, d.verifier, address(this));
        d.slash = new SCslash(d.committee, d.verifier, d.authreq, d.id, STAKE / 2, 20, address(this));
        d.checkpoint = new SCcheckpoint(d.committee, d.verifier);
        d.model = new SCmodel(d.score, keccak256("model-0"), 50, address(this));
        d.committee.setRoles(address(d.slash), address(d.checkpoint), IScore(address(d.score)));
        d.verifier.setAuthResp(IAuthResp(address(d.authresp)));
        uint256[2] memory g1 = [uint256(1), 2];
        uint256[4] memory g2 = Groth16.g2Generator();
        d.authresp.configure(IModel(address(d.model)), ISlash(address(d.slash)), g1, g2, g2, g2, vkIc(4));
        d.score.configure(ISlashTraining(address(d.slash)), g1, g2, g2, g2, vkIc(4));
        d.model.configure(ISlashAggregation(address(d.slash)), g1, g2, g2, g2, vkIc(4));
        d.slash.setCallers(address(d.authresp), address(d.score), address(d.model));
    }

    function validatorAddr(uint256 i) internal pure returns (address) {
        return vm.addr(0xA000 + i);
    }

    function validatorSk(uint256 i) internal pure returns (uint256) {
        return 0xB000 + i;
    }

    function registerValidators(Deployment memory d, uint256 n) internal {
        for (uint256 i = 0; i < n; i++) {
            address a = validatorAddr(i);
            bytes32 did = keccak256(abi.encodePacked("validator", i));
            d.id.register(did, a, keccak256("vc"), keccak256("cert"));
            vm.deal(a, 2 * STAKE);
            bytes memory pk = BLSHelper.publicKey(validatorSk(i));
            vm.prank(a);
            d.committee.register{value: STAKE}(pk, did);
        }
    }

    function runElection(Deployment memory d, uint256 n) internal returns (uint256 gasVrf, uint256 gasFinalize) {
        bytes32 seed = d.committee.seed();
        for (uint256 i = 0; i < n; i++) {
            bytes memory sigma = BLSHelper.sign(validatorSk(i), seed);
            vm.prank(validatorAddr(i));
            uint256 g = gasleft();
            d.committee.submitVrf(sigma);
            gasVrf += g - gasleft();
        }
        gasVrf /= n;
        vm.roll(block.number + 10);
        uint256 g2 = gasleft();
        d.committee.finalizeElection();
        gasFinalize = g2 - gasleft();
    }

    function indexOf(address[] memory members, address a) internal pure returns (uint256) {
        for (uint256 i = 0; i < members.length; i++) if (members[i] == a) return i;
        revert("missing");
    }

    function skOf(address a, uint256 n) internal pure returns (uint256) {
        for (uint256 i = 0; i < n; i++) if (validatorAddr(i) == a) return validatorSk(i);
        revert("sk");
    }

    function quorumSign(Deployment memory d, uint64 epoch, bytes32 message, uint256 n)
        internal
        view
        returns (bytes memory sig, uint256 bitmap)
    {
        address[] memory members = d.committee.committee(epoch);
        uint256 q = 2 * ((members.length - 1) / 3) + 1;
        bytes[] memory sigs = new bytes[](q);
        for (uint256 i = 0; i < q; i++) {
            sigs[i] = BLSHelper.sign(skOf(members[i], n), message);
            bitmap |= (uint256(1) << i);
        }
        sig = BLSHelper.aggregateSignatures(sigs);
    }

    function submitRequests(Deployment memory d, uint256 m) internal returns (uint256 gasPerRequest, uint256 batchId) {
        batchId = d.authreq.currentBatch();
        uint256 total;
        for (uint256 i = 0; i < m; i++) {
            uint256 dk = 0xD000 + i;
            bytes32 did = keccak256(abi.encodePacked("device", i, batchId));
            if (d.id.keyOf(did) == address(0)) d.id.register(did, vm.addr(dk), keccak256("vc"), keccak256("cert"));
            bytes32 ctHash = keccak256(abi.encodePacked("ciphertext", i));
            bytes32 nonce = keccak256(abi.encodePacked("nonce", i, batchId));
            uint64 ts = uint64(block.timestamp);
            bytes32 digest = d.authreq.requestDigest(did, ctHash, nonce, ts);
            (uint8 v, bytes32 r, bytes32 s) = vm.sign(dk, keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", digest)));
            uint256 g = gasleft();
            d.authreq.submitRequest(did, ctHash, nonce, ts, v, r, s);
            total += g - gasleft();
        }
        gasPerRequest = total / m;
        (,,, bool closed) = d.authreq.batchInfo(batchId);
        if (!closed) {
            vm.warp(block.timestamp + 3);
            d.authreq.closeBatch();
        }
    }

    function testFullEpochGas() public {
        uint256[4] memory ks = [uint256(4), 7, 13, 31];
        uint256[3] memory ms = [uint256(10), 50, 100];
        for (uint256 a = 0; a < ks.length; a++) {
            uint256 k = ks[a];
            uint256 n = 3 * k;
            Deployment memory d = deploy(k, false);
            registerValidators(d, n);
            d.committee.startEpoch(bytes32(0));
            (uint256 gVrf, uint256 gFin) = runElection(d, n);
            uint64 epoch = d.committee.epoch();
            address leader = d.committee.leaderOf(epoch);
            console2.log("K", k, "n", n);
            console2.log("  submitVrf gas (per validator)", gVrf);
            console2.log("  finalizeElection gas", gFin);
            uint256[] memory batchIds = new uint256[](ms.length);
            for (uint256 b = 0; b < ms.length; b++) {
                uint256 m = ms[b];
                (uint256 gReq, uint256 batchId) = submitRequests(d, m);
                batchIds[b] = batchId;
                bytes32 root = keccak256(abi.encodePacked("root", batchId));
                vm.prank(leader);
                uint256 g = gasleft();
                d.authresp.submitEvaluation(batchId, root, dummyProof());
                uint256 gEval = g - gasleft();
                (bytes memory sig, uint256 bitmap) = quorumSign(d, epoch, d.verifier.batchMessage(root, epoch, batchId), n);
                g = gasleft();
                d.verifier.finalize(batchId, sig, bitmap);
                uint256 gFinalize = g - gasleft();
                int256[] memory masked = new int256[](m);
                for (uint256 i = 0; i < m; i++) masked[i] = int256(uint256(keccak256(abi.encodePacked(i)))) >> 150;
                (bytes memory dsig, uint256 dbit) = quorumSign(d, epoch, d.authresp.decryptionMessage(batchId, masked), n);
                g = gasleft();
                d.authresp.postResults(batchId, masked, dsig, dbit);
                uint256 gPost = g - gasleft();
                console2.log("  M", m);
                console2.log("    submitRequest gas (per request)", gReq);
                console2.log("    submitEvaluation gas (attestation mode)", gEval);
                console2.log("    SCverify.finalize gas (BLS + bitmap)", gFinalize);
                console2.log("    postResults gas (masked values + quorum)", gPost);
            }
            {
                bytes32 h = d.checkpoint.checkpointHash(epoch, batchIds);
                (bytes memory csig, uint256 cbit) = quorumSign(d, epoch, h, n);
                uint256 g = gasleft();
                d.checkpoint.anchor(epoch, batchIds, csig, cbit);
                console2.log("  checkpoint anchor gas (3 batches)", g - gasleft());
            }
        }
    }

    function testScoresAndSlashingGas() public {
        uint256 k = 7;
        uint256 n = 21;
        Deployment memory d = deploy(k, true);
        registerValidators(d, n);
        d.committee.startEpoch(bytes32(0));
        runElection(d, n);
        uint64 epoch = d.committee.epoch();
        address leader = d.committee.leaderOf(epoch);
        address[] memory members = d.committee.committee(epoch);
        uint16[] memory scores = new uint16[](members.length);
        for (uint256 i = 0; i < members.length; i++) scores[i] = uint16(70 + i);
        (bytes memory sig, uint256 bitmap) = quorumSign(d, epoch, d.score.scoresMessage(epoch, members, scores), n);
        uint256 g = gasleft();
        d.score.commitScores(epoch, members, scores, sig, bitmap);
        console2.log("commitScores gas (K=7)", g - gasleft());
        (, uint256 batchId) = submitRequests(d, 10);
        vm.prank(leader);
        g = gasleft();
        bool ok = d.authresp.submitEvaluation(batchId, keccak256("forged"), dummyProof());
        console2.log("submitEvaluation with invalid proof (verify + slash) gas", g - gasleft());
        assertFalse(ok);
        (,, uint256 stakeAfter,,,,,,,) = d.committee.info(leader);
        assertEq(stakeAfter, STAKE / 2);
        address v = members[0];
        uint256 sk = skOf(v, n);
        bytes32 ra = keccak256("rootA");
        bytes32 rb = keccak256("rootB");
        bytes memory sa = BLSHelper.sign(sk, d.verifier.batchMessage(ra, epoch, 99));
        bytes memory sb = BLSHelper.sign(sk, d.verifier.batchMessage(rb, epoch, 99));
        g = gasleft();
        d.slash.reportDoubleSign(v, epoch, 99, ra, sa, rb, sb);
        console2.log("reportDoubleSign gas", g - gasleft());
        vm.expectRevert(bytes("settled"));
        d.slash.reportDoubleSign(v, epoch, 99, ra, sa, rb, sb);
        address m0 = members[1];
        vm.prank(m0);
        vm.expectRevert(bytes("leader"));
        d.score.submitTraining(1, keccak256("g"), keccak256("ds"), 9000, 100, dummyProof());
    }

    function testIdentityGas() public {
        SCid id = new SCid(address(this), address(this), keccak256("ca"));
        uint256 g = gasleft();
        id.register(keccak256("d1"), address(1), keccak256("vc"), keccak256("cert"));
        console2.log("SCid.register gas", g - gasleft());
        g = gasleft();
        id.revoke(keccak256("d1"));
        console2.log("SCid.revoke gas", g - gasleft());
        g = gasleft();
        bool valid = id.isValid(keccak256("d1"));
        console2.log("SCid.isValid gas", g - gasleft());
        assertFalse(valid);
    }

    function testGroth16VerifyGas() public {
        Groth16Harness h = new Groth16Harness(4);
        uint256[] memory inputs = new uint256[](4);
        for (uint256 i = 0; i < 4; i++) inputs[i] = uint256(keccak256(abi.encodePacked(i))) % Groth16.R;
        uint256 g = gasleft();
        h.verify(dummyProof(), inputs);
        console2.log("Groth16 verify gas (4 public inputs)", g - gasleft());
    }

    function testReplayAndClockRejected() public {
        Deployment memory d = deploy(4, false);
        registerValidators(d, 12);
        d.committee.startEpoch(bytes32(0));
        uint256 dk = 0xE001;
        bytes32 did = keccak256("dev");
        d.id.register(did, vm.addr(dk), keccak256("vc"), keccak256("cert"));
        bytes32 nonce = keccak256("n");
        uint64 ts = uint64(block.timestamp);
        bytes32 digest = d.authreq.requestDigest(did, keccak256("ct"), nonce, ts);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(dk, keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", digest)));
        d.authreq.submitRequest(did, keccak256("ct"), nonce, ts, v, r, s);
        vm.expectRevert(bytes("replay"));
        d.authreq.submitRequest(did, keccak256("ct"), nonce, ts, v, r, s);
        vm.warp(block.timestamp + 100);
        bytes32 nonce2 = keccak256("n2");
        bytes32 digest2 = d.authreq.requestDigest(did, keccak256("ct"), nonce2, ts);
        (v, r, s) = vm.sign(dk, keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", digest2)));
        vm.expectRevert(bytes("clock"));
        d.authreq.submitRequest(did, keccak256("ct"), nonce2, ts, v, r, s);
        d.id.revoke(did);
        vm.expectRevert(bytes("did"));
        d.authreq.submitRequest(did, keccak256("ct"), nonce2, uint64(block.timestamp), v, r, s);
    }

    function testRotationAndNoReelectedLeader() public {
        Deployment memory d = deploy(4, false);
        registerValidators(d, 12);
        d.committee.startEpoch(bytes32(0));
        runElection(d, 12);
        address l1 = d.committee.leaderOf(1);
        address[] memory c1 = d.committee.committee(1);
        d.committee.startEpoch(keccak256("ckpt1"));
        runElection(d, 12);
        address[] memory c2 = d.committee.committee(2);
        assertTrue(d.committee.leaderOf(2) != l1);
        uint256 overlap;
        for (uint256 i = 0; i < c1.length; i++) for (uint256 j = 0; j < c2.length; j++) if (c1[i] == c2[j]) overlap++;
        console2.log("committee overlap epoch1->2 (n=3K)", overlap);
    }

    function testInvalidVrfIsSlashed() public {
        Deployment memory d = deploy(4, false);
        registerValidators(d, 12);
        d.committee.startEpoch(bytes32(0));
        bytes memory bad = BLSHelper.sign(validatorSk(1), d.committee.seed());
        vm.prank(validatorAddr(0));
        d.committee.submitVrf(bad);
        (,, uint256 stake,,,, uint64 bannedUntil,,,) = d.committee.info(validatorAddr(0));
        assertEq(stake, STAKE - STAKE / 10);
        assertGt(bannedUntil, d.committee.epoch());
    }

    function testRevokedProcessingIsSlashed() public {
        Deployment memory d = deploy(4, false);
        registerValidators(d, 12);
        d.committee.startEpoch(bytes32(0));
        runElection(d, 12);
        uint64 epoch = d.committee.epoch();
        address leader = d.committee.leaderOf(epoch);
        uint256 batchId = d.authreq.currentBatch();
        uint256 dk = 0xE100;
        bytes32 did = keccak256("revoked-device");
        d.id.register(did, vm.addr(dk), keccak256("vc"), keccak256("cert"));
        bytes32 ct = keccak256("ct");
        bytes32 nonce = keccak256("nonce-r");
        uint64 ts = uint64(block.timestamp);
        bytes32 digest = d.authreq.requestDigest(did, ct, nonce, ts);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(dk, keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", digest)));
        bytes32 requestId = d.authreq.submitRequest(did, ct, nonce, ts, v, r, s);
        d.id.revoke(did);
        vm.warp(block.timestamp + 3);
        d.authreq.closeBatch();
        bytes32 root = keccak256("root-r");
        vm.prank(leader);
        d.authresp.submitEvaluation(batchId, root, dummyProof());
        (bytes memory sig, uint256 bitmap) = quorumSign(d, epoch, d.verifier.batchMessage(root, epoch, batchId), 12);
        d.verifier.finalize(batchId, sig, bitmap);
        d.slash.reportRevokedProcessing(requestId);
        (,, uint256 stake,,,,,,,) = d.committee.info(leader);
        assertEq(stake, STAKE / 2);
    }
}
