// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Groth16} from "./Groth16.sol";
import {SCauthreq} from "./SCauthreq.sol";
import {SCcommittee} from "./SCcommittee.sol";
import {SCverify} from "./SCverify.sol";

interface ISlash {
    function reportInvalidEvaluation(address leader, uint256 batchId) external;
}

interface IModel {
    function modelHash() external view returns (bytes32);
}

contract SCauthresp {
    using Groth16 for Groth16.VerifyingKey;

    SCauthreq public immutable authreq;
    SCcommittee public immutable committee;
    SCverify public immutable verifier;
    IModel public model;
    ISlash public slasher;
    address public governance;
    bool public immutable proofRequired;

    Groth16.VerifyingKey private vk;

    mapping(uint256 => bytes32) public verifiedRoot;
    mapping(uint256 => uint64) public batchEpoch;
    mapping(uint256 => bytes32) public resultHash;

    event EvaluationAccepted(uint256 indexed batchId, bytes32 root, address leader);
    event EvaluationRejected(uint256 indexed batchId, address leader);
    event MaskedResults(uint256 indexed batchId, int256[] masked);

    constructor(SCauthreq authreq_, SCcommittee committee_, SCverify verifier_, bool proofRequired_, address governance_) {
        authreq = authreq_;
        committee = committee_;
        verifier = verifier_;
        proofRequired = proofRequired_;
        governance = governance_;
    }

    function configure(IModel model_, ISlash slasher_, uint256[2] calldata alpha, uint256[4] calldata beta, uint256[4] calldata gamma,
        uint256[4] calldata delta, uint256[2][] calldata ic) external
    {
        require(msg.sender == governance && address(slasher) == address(0), "once");
        model = model_;
        slasher = slasher_;
        vk.alpha = alpha;
        vk.beta = beta;
        vk.gamma = gamma;
        vk.delta = delta;
        for (uint256 i = 0; i < ic.length; i++) vk.ic.push(ic[i]);
    }

    function publicInputs(uint256 batchId, bytes32 root) public view returns (uint256[] memory inputs) {
        (bytes32 acc, uint64 epoch,,) = authreq.batchInfo(batchId);
        inputs = new uint256[](4);
        inputs[0] = uint256(acc) % Groth16.R;
        inputs[1] = uint256(root) % Groth16.R;
        inputs[2] = uint256(model.modelHash()) % Groth16.R;
        inputs[3] = epoch;
    }

    function submitEvaluation(uint256 batchId, bytes32 root, Groth16.Proof calldata proof) external returns (bool) {
        (, uint64 epoch,, bool closed) = authreq.batchInfo(batchId);
        require(closed, "open");
        require(committee.leaderOf(epoch) == msg.sender, "leader");
        require(verifiedRoot[batchId] == bytes32(0), "done");
        if (proofRequired && !vk.verify(proof, publicInputs(batchId, root))) {
            emit EvaluationRejected(batchId, msg.sender);
            slasher.reportInvalidEvaluation(msg.sender, batchId);
            return false;
        }
        verifiedRoot[batchId] = root;
        batchEpoch[batchId] = epoch;
        emit EvaluationAccepted(batchId, root, msg.sender);
        return true;
    }

    function decryptionMessage(uint256 batchId, int256[] calldata masked) public pure returns (bytes32) {
        return keccak256(abi.encodePacked("IoTAuth/decrypt", batchId, masked));
    }

    function postResults(uint256 batchId, int256[] calldata masked, bytes calldata signature, uint256 bitmap) external {
        require(verifier.finalizedRoot(batchId) != bytes32(0), "consensus");
        require(resultHash[batchId] == bytes32(0), "posted");
        (,, uint32 count,) = authreq.batchInfo(batchId);
        require(masked.length == count, "length");
        bytes32 m = decryptionMessage(batchId, masked);
        require(verifier.checkQuorum(batchEpoch[batchId], m, signature, bitmap), "quorum");
        resultHash[batchId] = m;
        emit MaskedResults(batchId, masked);
    }
}
