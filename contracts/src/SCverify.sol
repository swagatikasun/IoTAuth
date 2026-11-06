// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BLS} from "./BLS.sol";
import {SCcommittee} from "./SCcommittee.sol";

interface IAuthResp {
    function verifiedRoot(uint256 batchId) external view returns (bytes32);
    function batchEpoch(uint256 batchId) external view returns (uint64);
}

contract SCverify {
    SCcommittee public immutable committee;
    IAuthResp public authResp;
    address public governance;

    mapping(uint256 => bytes32) public finalizedRoot;
    mapping(uint256 => uint256) public finalizedBitmap;

    event ConsensusReached(uint256 indexed batchId, bytes32 root, uint256 bitmap, uint64 epoch);

    constructor(SCcommittee committee_, address governance_) {
        committee = committee_;
        governance = governance_;
    }

    function setAuthResp(IAuthResp a) external {
        require(msg.sender == governance && address(authResp) == address(0), "once");
        authResp = a;
    }

    function popcount(uint256 x) internal pure returns (uint256 c) {
        while (x != 0) {
            x &= x - 1;
            c++;
        }
    }

    function aggregateKey(uint64 epoch, uint256 bitmap) public view returns (bytes memory acc) {
        address[] memory members = committee.committee(epoch);
        bool first = true;
        for (uint256 i = 0; i < members.length; i++) {
            if ((bitmap >> i) & 1 == 1) {
                bytes memory pk = committee.publicKey(members[i]);
                acc = first ? pk : BLS.addG1(acc, pk);
                first = false;
            }
        }
        require(!first, "empty");
    }

    function checkQuorum(uint64 epoch, bytes32 message, bytes calldata signature, uint256 bitmap) public view returns (bool) {
        address[] memory members = committee.committee(epoch);
        require(members.length > 0, "no committee");
        require(bitmap >> members.length == 0, "bitmap");
        uint256 f = (members.length - 1) / 3;
        if (popcount(bitmap) < 2 * f + 1) return false;
        return BLS.verify(aggregateKey(epoch, bitmap), message, signature);
    }

    function batchMessage(bytes32 root, uint64 epoch, uint256 batchId) public pure returns (bytes32) {
        return keccak256(abi.encodePacked("IoTAuth/batch", root, epoch, batchId));
    }

    function finalize(uint256 batchId, bytes calldata signature, uint256 bitmap) external {
        require(finalizedRoot[batchId] == bytes32(0), "final");
        bytes32 root = authResp.verifiedRoot(batchId);
        require(root != bytes32(0), "unverified");
        uint64 epoch = authResp.batchEpoch(batchId);
        require(checkQuorum(epoch, batchMessage(root, epoch, batchId), signature, bitmap), "quorum");
        finalizedRoot[batchId] = root;
        finalizedBitmap[batchId] = bitmap;
        emit ConsensusReached(batchId, root, bitmap, epoch);
    }
}
