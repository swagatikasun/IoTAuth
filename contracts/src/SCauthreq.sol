// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {SCid} from "./SCid.sol";
import {SCcommittee} from "./SCcommittee.sol";

contract SCauthreq {
    struct Batch {
        bytes32 accumulator;
        uint64 epoch;
        uint64 openedAt;
        uint32 count;
        bool closed;
    }

    struct Request {
        bytes32 did;
        uint256 batchId;
        uint32 index;
    }

    SCid public immutable id;
    SCcommittee public immutable committee;
    uint32 public immutable maxBatch;
    uint64 public immutable batchTimeout;
    uint64 public immutable clockTolerance;

    uint256 public currentBatch;
    mapping(uint256 => Batch) public batches;
    mapping(bytes32 => Request) public requests;
    mapping(uint64 => mapping(bytes32 => bool)) public nonceUsed;

    event RequestLogged(bytes32 indexed requestId, bytes32 indexed did, bytes32 ciphertextHash, uint256 batchId, uint32 index);
    event BatchClosed(uint256 indexed batchId, bytes32 accumulator, uint32 count, uint64 epoch);

    constructor(SCid id_, SCcommittee committee_, uint32 maxBatch_, uint64 batchTimeout_, uint64 clockTolerance_) {
        id = id_;
        committee = committee_;
        maxBatch = maxBatch_;
        batchTimeout = batchTimeout_;
        clockTolerance = clockTolerance_;
        currentBatch = 1;
    }

    function requestDigest(bytes32 did, bytes32 ciphertextHash, bytes32 nonce, uint64 timestamp) public view returns (bytes32) {
        return keccak256(abi.encodePacked("IoTAuth/request", block.chainid, address(this), did, ciphertextHash, nonce, timestamp));
    }

    function submitRequest(bytes32 did, bytes32 ciphertextHash, bytes32 nonce, uint64 timestamp, uint8 v, bytes32 r, bytes32 s)
        external
        returns (bytes32 requestId)
    {
        require(id.isValid(did), "did");
        uint64 epoch = committee.epoch();
        require(!nonceUsed[epoch][nonce], "replay");
        uint64 nowTs = uint64(block.timestamp);
        require(timestamp + clockTolerance >= nowTs && timestamp <= nowTs + clockTolerance, "clock");
        bytes32 digest = requestDigest(did, ciphertextHash, nonce, timestamp);
        address signer = ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", digest)), v, r, s);
        require(signer != address(0) && signer == id.keyOf(did), "signature");
        nonceUsed[epoch][nonce] = true;
        _rollBatch(epoch);
        Batch storage b = batches[currentBatch];
        if (b.count == 0) {
            b.epoch = epoch;
            b.openedAt = nowTs;
        }
        requestId = digest;
        b.accumulator = keccak256(abi.encodePacked(b.accumulator, requestId, ciphertextHash));
        requests[requestId] = Request(did, currentBatch, b.count);
        emit RequestLogged(requestId, did, ciphertextHash, currentBatch, b.count);
        b.count += 1;
        if (b.count >= maxBatch) _close(currentBatch);
    }

    function closeBatch() external {
        Batch storage b = batches[currentBatch];
        require(b.count > 0, "empty");
        require(block.timestamp >= b.openedAt + batchTimeout || committee.epoch() != b.epoch, "early");
        _close(currentBatch);
    }

    function _rollBatch(uint64 epoch) internal {
        Batch storage b = batches[currentBatch];
        if (b.count > 0 && (b.epoch != epoch || block.timestamp >= b.openedAt + batchTimeout)) {
            _close(currentBatch);
        }
    }

    function _close(uint256 batchId) internal {
        Batch storage b = batches[batchId];
        b.closed = true;
        emit BatchClosed(batchId, b.accumulator, b.count, b.epoch);
        currentBatch = batchId + 1;
    }

    function batchInfo(uint256 batchId) external view returns (bytes32 accumulator, uint64 epoch, uint32 count, bool closed) {
        Batch storage b = batches[batchId];
        return (b.accumulator, b.epoch, b.count, b.closed);
    }
}
