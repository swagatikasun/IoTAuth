// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {SCcommittee} from "./SCcommittee.sol";
import {SCverify} from "./SCverify.sol";

contract SCcheckpoint {
    SCcommittee public immutable committee;
    SCverify public immutable verifier;

    mapping(uint64 => bytes32) public checkpointOf;
    mapping(uint64 => bool) public deferred;

    event Anchored(uint64 indexed epoch, bytes32 checkpointHash, uint256 batches);

    constructor(SCcommittee committee_, SCverify verifier_) {
        committee = committee_;
        verifier = verifier_;
    }

    function checkpointHash(uint64 epoch, uint256[] calldata batchIds) public view returns (bytes32) {
        bytes32 roots;
        bytes32 bitmaps;
        for (uint256 i = 0; i < batchIds.length; i++) {
            bytes32 r = verifier.finalizedRoot(batchIds[i]);
            require(r != bytes32(0), "unfinalized");
            roots = keccak256(abi.encodePacked(roots, r));
            bitmaps = keccak256(abi.encodePacked(bitmaps, verifier.finalizedBitmap(batchIds[i])));
        }
        return keccak256(abi.encodePacked("IoTAuth/checkpoint", epoch, roots, bitmaps));
    }

    function anchor(uint64 epoch, uint256[] calldata batchIds, bytes calldata signature, uint256 bitmap) external {
        require(checkpointOf[epoch] == bytes32(0), "anchored");
        require(epoch == committee.epoch(), "epoch");
        bytes32 h = checkpointHash(epoch, batchIds);
        require(verifier.checkQuorum(epoch, h, signature, bitmap), "quorum");
        checkpointOf[epoch] = h;
        emit Anchored(epoch, h, batchIds.length);
        committee.startEpoch(h);
    }
}
