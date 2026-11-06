// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BLS} from "./BLS.sol";
import {SCid} from "./SCid.sol";

interface IScore {
    function leaderScore(address v, uint64 epoch) external view returns (uint256);
}

contract SCcommittee {
    struct Validator {
        bytes pk;
        bytes32 did;
        uint256 stake;
        int256 reputation;
        uint64 lastLeader;
        uint64 lastCommittee;
        uint64 bannedUntil;
        uint64 exitRequested;
        uint64 joined;
        bool active;
    }

    SCid public immutable id;
    uint256 public immutable committeeSize;
    uint256 public immutable minStake;
    uint64 public immutable banEpochs;
    bool public immutable uniformCommittee;
    address public governance;
    address public slasher;
    address public checkpointer;
    IScore public score;

    uint64 public epoch;
    bytes32 public seed;
    uint256 public electionDeadline;
    uint256 public immutable electionWindow;

    address[] public validators;
    mapping(address => Validator) public info;
    mapping(uint64 => address[]) private committees;
    mapping(uint64 => address) public leaderOf;
    mapping(uint64 => mapping(address => bool)) public submitted;

    address[] private topAddr;
    uint256[] private topScore;

    event Registered(address indexed v);
    event VrfAccepted(address indexed v, uint64 epoch, uint256 weighted);
    event CommitteeFormed(uint64 indexed epoch, address[] committee);
    event LeaderElected(address indexed leader, uint64 indexed epoch, bytes32 seed);
    event Penalized(address indexed v, uint256 stake, int256 reputation, uint64 bannedUntil);

    constructor(SCid id_, uint256 committeeSize_, uint256 minStake_, uint64 banEpochs_, bool uniform_, uint256 electionWindow_,
        address governance_)
    {
        electionWindow = electionWindow_;
        id = id_;
        committeeSize = committeeSize_;
        minStake = minStake_;
        banEpochs = banEpochs_;
        uniformCommittee = uniform_;
        governance = governance_;
    }

    function setRoles(address slasher_, address checkpointer_, IScore score_) external {
        require(msg.sender == governance && slasher == address(0), "once");
        slasher = slasher_;
        checkpointer = checkpointer_;
        score = score_;
    }

    function register(bytes calldata pk, bytes32 did) external payable {
        require(msg.value >= minStake, "stake");
        require(id.isValid(did) && id.keyOf(did) == msg.sender, "identity");
        Validator storage v = info[msg.sender];
        require(!v.active && v.exitRequested == 0, "registered");
        require(pk.length == 128, "pk");
        v.pk = pk;
        v.did = did;
        v.stake += msg.value;
        v.active = true;
        v.joined = epoch;
        validators.push(msg.sender);
        emit Registered(msg.sender);
    }

    function requestExit() external {
        Validator storage v = info[msg.sender];
        require(v.active, "inactive");
        v.active = false;
        v.exitRequested = epoch;
    }

    function withdraw() external {
        Validator storage v = info[msg.sender];
        require(v.exitRequested != 0 && epoch >= v.exitRequested + banEpochs + 3, "unbonding");
        require(epoch >= v.bannedUntil, "banned");
        uint256 amount = v.stake;
        v.stake = 0;
        v.exitRequested = 0;
        payable(msg.sender).transfer(amount);
    }

    function multiplier(address a) public view returns (uint256) {
        if (uniformCommittee) return 100;
        Validator storage v = info[a];
        if (v.lastLeader != 0 && v.lastLeader + 1 == epoch) return 5;
        if (v.lastCommittee != 0 && v.lastCommittee + 1 == epoch) return 30;
        uint64 since = v.lastCommittee > v.joined ? v.lastCommittee : v.joined;
        uint64 rested = epoch > since + 1 ? epoch - 1 - since : 0;
        if (rested <= 1) return 100;
        if (rested == 2) return 120;
        return 150;
    }

    function startEpoch(bytes32 checkpointHash) external {
        require(msg.sender == checkpointer || msg.sender == governance, "auth");
        epoch += 1;
        seed = keccak256(abi.encodePacked(blockhash(block.number - 1), checkpointHash, epoch));
        electionDeadline = block.number + electionWindow;
        delete topAddr;
        delete topScore;
    }

    function eligible(address a) public view returns (bool) {
        Validator storage v = info[a];
        return v.active && v.bannedUntil <= epoch && v.stake >= minStake;
    }

    function submitVrf(bytes calldata sigma) external {
        require(block.number <= electionDeadline, "closed");
        require(eligible(msg.sender) && !submitted[epoch][msg.sender], "ineligible");
        submitted[epoch][msg.sender] = true;
        if (!BLS.verify(info[msg.sender].pk, seed, sigma)) {
            _penalize(msg.sender, minStake / 10, 10);
            return;
        }
        uint256 y = uint256(keccak256(sigma)) >> 16;
        uint256 weighted = y * multiplier(msg.sender);
        _insertTop(msg.sender, weighted);
        emit VrfAccepted(msg.sender, epoch, weighted);
    }

    function _insertTop(address a, uint256 s) internal {
        uint256 n = topScore.length;
        if (n == committeeSize && s <= topScore[n - 1]) return;
        if (n < committeeSize) {
            topScore.push(s);
            topAddr.push(a);
            n += 1;
        } else {
            topScore[n - 1] = s;
            topAddr[n - 1] = a;
        }
        uint256 i = n - 1;
        while (i > 0 && topScore[i - 1] < s) {
            topScore[i] = topScore[i - 1];
            topAddr[i] = topAddr[i - 1];
            i--;
        }
        topScore[i] = s;
        topAddr[i] = a;
    }

    function finalizeElection() external returns (address leader) {
        require(block.number > electionDeadline, "open");
        require(committees[epoch].length == 0, "done");
        require(topAddr.length == committeeSize, "abort: too few eligible");
        address prev = epoch > 1 ? leaderOf[epoch - 1] : address(0);
        uint256 best;
        bool found;
        for (uint256 i = 0; i < topAddr.length; i++) {
            address a = topAddr[i];
            committees[epoch].push(a);
            info[a].lastCommittee = epoch;
            if (a == prev) continue;
            uint256 l = address(score) == address(0) ? 0 : score.leaderScore(a, epoch);
            if (!found || l > best) {
                best = l;
                leader = a;
                found = true;
            }
        }
        leaderOf[epoch] = leader;
        info[leader].lastLeader = epoch;
        emit CommitteeFormed(epoch, committees[epoch]);
        emit LeaderElected(leader, epoch, seed);
    }

    function committee(uint64 e) external view returns (address[] memory) {
        return committees[e];
    }

    function publicKey(address a) external view returns (bytes memory) {
        return info[a].pk;
    }

    function faultThreshold() public view returns (uint256) {
        return (committeeSize - 1) / 3;
    }

    function penalize(address a, uint256 stakePenalty, int256 reputationPenalty) external {
        require(msg.sender == slasher, "slasher");
        _penalize(a, stakePenalty, reputationPenalty);
    }

    function adjustReputation(address a, int256 delta) external {
        require(msg.sender == slasher || msg.sender == address(score), "auth");
        info[a].reputation += delta;
    }

    function _penalize(address a, uint256 stakePenalty, int256 reputationPenalty) internal {
        Validator storage v = info[a];
        uint256 cut = stakePenalty > v.stake ? v.stake : stakePenalty;
        v.stake -= cut;
        v.reputation -= reputationPenalty;
        v.bannedUntil = epoch + 1 + banEpochs;
        emit Penalized(a, cut, v.reputation, v.bannedUntil);
    }

    function validatorCount() external view returns (uint256) {
        return validators.length;
    }
}
