// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BLS} from "./BLS.sol";
import {SCauthreq} from "./SCauthreq.sol";
import {SCcommittee} from "./SCcommittee.sol";
import {SCid} from "./SCid.sol";
import {SCverify} from "./SCverify.sol";

contract SCslash {
    SCcommittee public immutable committee;
    SCverify public immutable verifier;
    SCauthreq public immutable authreq;
    SCid public immutable id;
    uint256 public immutable stakePenalty;
    int256 public immutable reputationPenalty;
    address public authresp;
    address public score;
    address public model;
    address public governance;

    mapping(bytes32 => bool) public settled;

    event Slashed(address indexed validator, uint8 condition, bytes32 evidence);

    constructor(SCcommittee committee_, SCverify verifier_, SCauthreq authreq_, SCid id_, uint256 stakePenalty_,
        int256 reputationPenalty_, address governance_)
    {
        committee = committee_;
        verifier = verifier_;
        authreq = authreq_;
        id = id_;
        stakePenalty = stakePenalty_;
        reputationPenalty = reputationPenalty_;
        governance = governance_;
    }

    function setCallers(address authresp_, address score_, address model_) external {
        require(msg.sender == governance && authresp == address(0), "once");
        authresp = authresp_;
        score = score_;
        model = model_;
    }

    function _slash(address v, uint8 condition, bytes32 evidence) internal {
        require(!settled[evidence], "settled");
        settled[evidence] = true;
        committee.penalize(v, stakePenalty, reputationPenalty);
        emit Slashed(v, condition, evidence);
    }

    function reportInvalidEvaluation(address leader, uint256 batchId) external {
        require(msg.sender == authresp, "auth");
        _slash(leader, 2, keccak256(abi.encodePacked("eval", batchId, leader)));
    }

    function reportInvalidTraining(address leader, uint256 round) external {
        require(msg.sender == score, "auth");
        _slash(leader, 3, keccak256(abi.encodePacked("train", round, leader)));
    }

    function reportInvalidAggregation(address aggregator, uint256 round) external {
        require(msg.sender == model, "auth");
        _slash(aggregator, 3, keccak256(abi.encodePacked("agg", round, aggregator)));
    }

    function reportRevokedProcessing(bytes32 requestId) external {
        (bytes32 did, uint256 batchId,) = authreq.requests(requestId);
        require(did != bytes32(0), "request");
        (, uint64 epoch, uint64 openedAt,,) = authreq.batches(batchId);
        uint64 revoked = id.revokedAt(did);
        require(revoked != 0 && revoked <= openedAt + authreq.batchTimeout(), "not revoked in time");
        require(verifier.finalizedRoot(batchId) != bytes32(0), "not processed");
        _slash(committee.leaderOf(epoch), 4, keccak256(abi.encodePacked("revoked", requestId)));
    }

    function reportDoubleSign(address v, uint64 epoch, uint256 batchId, bytes32 rootA, bytes calldata sigA, bytes32 rootB,
        bytes calldata sigB) external
    {
        require(rootA != rootB, "same");
        bytes memory pk = committee.publicKey(v);
        require(BLS.verify(pk, verifier.batchMessage(rootA, epoch, batchId), sigA), "sigA");
        require(BLS.verify(pk, verifier.batchMessage(rootB, epoch, batchId), sigB), "sigB");
        _slash(v, 5, keccak256(abi.encodePacked("double", v, epoch, batchId)));
    }
}
