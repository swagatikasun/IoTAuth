// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Groth16} from "./Groth16.sol";
import {SCscore} from "./SCscore.sol";

interface ISlashAggregation {
    function reportInvalidAggregation(address aggregator, uint256 round) external;
}

contract SCmodel {
    using Groth16 for Groth16.VerifyingKey;

    SCscore public immutable score;
    address public governance;
    ISlashAggregation public slasher;
    Groth16.VerifyingKey private aggVk;

    bytes32 public modelHash;
    uint256 public modelAccuracyBps;
    uint256 public round;
    uint256 public immutable toleranceBps;
    int256[3] public thresholds;

    event ModelUpdated(uint256 indexed round, bytes32 modelHash, uint256 accuracyBps);
    event AggregationRejected(uint256 indexed round, address aggregator, uint8 reason);
    event ThresholdsSet(int256 theta1, int256 theta2, int256 theta3);

    constructor(SCscore score_, bytes32 initialModel, uint256 toleranceBps_, address governance_) {
        score = score_;
        modelHash = initialModel;
        toleranceBps = toleranceBps_;
        governance = governance_;
    }

    function configure(ISlashAggregation slasher_, uint256[2] calldata alpha, uint256[4] calldata beta, uint256[4] calldata gamma,
        uint256[4] calldata delta, uint256[2][] calldata ic) external
    {
        require(msg.sender == governance && address(slasher) == address(0), "once");
        slasher = slasher_;
        aggVk.alpha = alpha;
        aggVk.beta = beta;
        aggVk.gamma = gamma;
        aggVk.delta = delta;
        for (uint256 i = 0; i < ic.length; i++) aggVk.ic.push(ic[i]);
    }

    function setThresholds(int256 t1, int256 t2, int256 t3) external {
        require(msg.sender == governance, "gov");
        require(t1 > t2 && t2 > t3, "order");
        thresholds = [t1, t2, t3];
        emit ThresholdsSet(t1, t2, t3);
    }

    function submitAggregate(uint256 r, bytes32 newModelHash, address[] calldata leaders, uint256 accuracyBps,
        Groth16.Proof calldata proof) external returns (bool)
    {
        require(r == round + 1, "round");
        bytes32 grads;
        for (uint256 i = 0; i < leaders.length; i++) {
            bytes32 g = score.gradientHash(r, leaders[i]);
            if (g == bytes32(0)) return _reject(r, 1);
            grads = keccak256(abi.encodePacked(grads, g));
        }
        if (accuracyBps + toleranceBps < modelAccuracyBps) return _reject(r, 2);
        uint256[] memory inputs = new uint256[](4);
        inputs[0] = uint256(grads) % Groth16.R;
        inputs[1] = uint256(newModelHash) % Groth16.R;
        inputs[2] = accuracyBps;
        inputs[3] = r;
        if (!aggVk.verify(proof, inputs)) return _reject(r, 3);
        round = r;
        modelHash = newModelHash;
        modelAccuracyBps = accuracyBps;
        emit ModelUpdated(r, newModelHash, accuracyBps);
        return true;
    }

    function _reject(uint256 r, uint8 reason) internal returns (bool) {
        emit AggregationRejected(r, msg.sender, reason);
        slasher.reportInvalidAggregation(msg.sender, r);
        return false;
    }
}
