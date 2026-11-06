// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script, console2} from "forge-std/Script.sol";
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
import {BLSSigner} from "../src/test/BLSSigner.sol";

contract Deploy is Script {
    function ic() internal pure returns (uint256[2][] memory out) {
        out = new uint256[2][](5);
        for (uint256 i = 0; i < 5; i++) out[i] = [uint256(1), 2];
    }

    function run() external {
        uint256 k = vm.envUint("COMMITTEE_SIZE");
        bool proofRequired = vm.envBool("PROOF_REQUIRED");
        uint32 maxBatch = uint32(vm.envUint("MAX_BATCH"));
        address me = vm.addr(vm.envUint("DEPLOYER_KEY"));
        vm.startBroadcast(vm.envUint("DEPLOYER_KEY"));
        SCid id = new SCid(me, me, keccak256("ca-key"));
        SCcommittee committee = new SCcommittee(id, k, 1 ether, 5, false, vm.envUint("ELECTION_WINDOW"), me);
        SCverify verifier = new SCverify(committee, me);
        SCauthreq authreq = new SCauthreq(id, committee, maxBatch, 3600, 600);
        SCauthresp authresp = new SCauthresp(authreq, committee, verifier, proofRequired, me);
        SCscore score = new SCscore(committee, verifier, me);
        SCslash slash = new SCslash(committee, verifier, authreq, id, 0.5 ether, 20, me);
        SCcheckpoint checkpoint = new SCcheckpoint(committee, verifier);
        SCmodel model = new SCmodel(score, keccak256("model-0"), 50, me);
        BLSSigner signer = new BLSSigner();
        committee.setRoles(address(slash), address(checkpoint), IScore(address(score)));
        verifier.setAuthResp(IAuthResp(address(authresp)));
        uint256[2] memory g1 = [uint256(1), 2];
        uint256[4] memory g2 = Groth16.g2Generator();
        authresp.configure(IModel(address(model)), ISlash(address(slash)), g1, g2, g2, g2, ic());
        score.configure(ISlashTraining(address(slash)), g1, g2, g2, g2, ic());
        model.configure(ISlashAggregation(address(slash)), g1, g2, g2, g2, ic());
        slash.setCallers(address(authresp), address(score), address(model));
        vm.stopBroadcast();
        string memory j = "deploy";
        vm.serializeAddress(j, "id", address(id));
        vm.serializeAddress(j, "committee", address(committee));
        vm.serializeAddress(j, "verifier", address(verifier));
        vm.serializeAddress(j, "authreq", address(authreq));
        vm.serializeAddress(j, "authresp", address(authresp));
        vm.serializeAddress(j, "score", address(score));
        vm.serializeAddress(j, "slash", address(slash));
        vm.serializeAddress(j, "checkpoint", address(checkpoint));
        vm.serializeAddress(j, "model", address(model));
        string memory out = vm.serializeAddress(j, "signer", address(signer));
        vm.writeJson(out, vm.envString("DEPLOY_OUT"));
        console2.log("deployed");
    }
}
