// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Groth16} from "./Groth16.sol";
import {SCcommittee} from "./SCcommittee.sol";
import {SCverify} from "./SCverify.sol";

interface ISlashTraining {
    function reportInvalidTraining(address leader, uint256 round) external;
}

contract SCscore {
    using Groth16 for Groth16.VerifyingKey;

    struct Record {
        uint16[3] recent;
        uint8 filled;
        uint8 head;
        uint64 creditEpoch;
        uint256 creditScaled;
        uint64 lastLed;
        bool everLed;
    }

    uint256 public constant SCALE = 1e6;
    uint256 public constant DECAY = 860000;
    uint256 public constant CREDIT = 10 * SCALE;
    uint256 public constant W_PERF = 6;
    uint256 public constant W_CRED = 3;
    uint256 public constant W_FAIR = 1;

    SCcommittee public immutable committee;
    SCverify public immutable verifier;
    ISlashTraining public slasher;
    address public governance;
    Groth16.VerifyingKey private trainVk;

    mapping(address => Record) public records;
    mapping(uint256 => mapping(address => bytes32)) public gradientHash;
    mapping(uint256 => mapping(address => uint256)) public gradientAccuracyBps;

    event ScoresCommitted(uint64 indexed epoch, uint256 count);
    event TrainingAccepted(uint256 indexed round, address indexed leader, bytes32 gradientHash, uint256 accuracyBps);
    event TrainingRejected(uint256 indexed round, address indexed leader);

    constructor(SCcommittee committee_, SCverify verifier_, address governance_) {
        committee = committee_;
        verifier = verifier_;
        governance = governance_;
    }

    function configure(ISlashTraining slasher_, uint256[2] calldata alpha, uint256[4] calldata beta, uint256[4] calldata gamma,
        uint256[4] calldata delta, uint256[2][] calldata ic) external
    {
        require(msg.sender == governance && address(slasher) == address(0), "once");
        slasher = slasher_;
        trainVk.alpha = alpha;
        trainVk.beta = beta;
        trainVk.gamma = gamma;
        trainVk.delta = delta;
        for (uint256 i = 0; i < ic.length; i++) trainVk.ic.push(ic[i]);
    }

    function scoresMessage(uint64 epoch, address[] calldata vals, uint16[] calldata scores) public pure returns (bytes32) {
        return keccak256(abi.encodePacked("IoTAuth/scores", epoch, vals, scores));
    }

    function commitScores(uint64 epoch, address[] calldata vals, uint16[] calldata scores, bytes calldata signature, uint256 bitmap)
        external
    {
        require(vals.length == scores.length, "length");
        require(verifier.checkQuorum(epoch, scoresMessage(epoch, vals, scores), signature, bitmap), "quorum");
        address leader = committee.leaderOf(epoch);
        for (uint256 i = 0; i < vals.length; i++) {
            require(scores[i] <= 100, "range");
            Record storage r = records[vals[i]];
            r.recent[r.head] = scores[i];
            r.head = uint8((r.head + 1) % 3);
            if (r.filled < 3) r.filled++;
            if (vals[i] == leader) {
                _decay(r, epoch);
                if (scores[i] >= 70) r.creditScaled += CREDIT;
                r.lastLed = epoch;
                r.everLed = true;
            }
        }
        emit ScoresCommitted(epoch, vals.length);
    }

    function _decay(Record storage r, uint64 epoch) internal {
        uint256 c = r.creditScaled;
        for (uint64 e = r.creditEpoch; e < epoch && c > 0; e++) {
            c = (c * DECAY) / SCALE;
        }
        r.creditScaled = c;
        r.creditEpoch = epoch;
    }

    function credits(address v, uint64 epoch) public view returns (uint256 c) {
        Record storage r = records[v];
        c = r.creditScaled;
        for (uint64 e = r.creditEpoch; e < epoch && c > 0; e++) {
            c = (c * DECAY) / SCALE;
        }
    }

    function performance(address v) public view returns (uint256) {
        Record storage r = records[v];
        if (r.filled == 0) return 0;
        uint256 sum;
        for (uint256 i = 0; i < r.filled; i++) sum += r.recent[i];
        return (sum * SCALE) / r.filled;
    }

    function fairness(address v, uint64 epoch) public view returns (uint256) {
        Record storage r = records[v];
        if (!r.everLed) return 10 * SCALE;
        if (epoch >= r.lastLed + 5) return 5 * SCALE;
        return 0;
    }

    function leaderScore(address v, uint64 epoch) external view returns (uint256) {
        return W_PERF * performance(v) + W_CRED * credits(v, epoch) + W_FAIR * fairness(v, epoch);
    }

    function submitTraining(uint256 round, bytes32 gradHash, bytes32 datasetRoot, uint256 accuracyBps, uint256 normBound,
        Groth16.Proof calldata proof) external returns (bool)
    {
        uint64 epoch = committee.epoch();
        require(committee.leaderOf(epoch) == msg.sender, "leader");
        uint256[] memory inputs = new uint256[](4);
        inputs[0] = uint256(gradHash) % Groth16.R;
        inputs[1] = uint256(datasetRoot) % Groth16.R;
        inputs[2] = accuracyBps;
        inputs[3] = normBound;
        if (!trainVk.verify(proof, inputs)) {
            emit TrainingRejected(round, msg.sender);
            slasher.reportInvalidTraining(msg.sender, round);
            return false;
        }
        gradientHash[round][msg.sender] = gradHash;
        gradientAccuracyBps[round][msg.sender] = accuracyBps;
        emit TrainingAccepted(round, msg.sender, gradHash, accuracyBps);
        return true;
    }
}
