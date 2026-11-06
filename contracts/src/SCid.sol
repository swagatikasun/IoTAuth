// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract SCid {
    struct Identity {
        address key;
        bytes32 vcHash;
        bytes32 certHash;
        uint64 registeredAt;
        uint64 revokedAt;
    }

    address public governance;
    address public ca;
    bytes32 public caKeyHash;
    bytes32 public heKeyHash;
    bytes32 public previousHeKeyHash;
    uint64 public heKeyEpoch;
    uint64 public heGraceUntilEpoch;

    mapping(bytes32 => Identity) private identities;

    event Registered(bytes32 indexed did, address key, bytes32 vcHash);
    event Revoked(bytes32 indexed did, uint64 at);
    event CaRotated(address ca, bytes32 caKeyHash);
    event HeKeyAnnounced(bytes32 keyHash, uint64 epoch, uint64 graceUntil);

    modifier onlyCa() {
        require(msg.sender == ca, "ca");
        _;
    }

    modifier onlyGovernance() {
        require(msg.sender == governance, "gov");
        _;
    }

    constructor(address governance_, address ca_, bytes32 caKeyHash_) {
        governance = governance_;
        ca = ca_;
        caKeyHash = caKeyHash_;
    }

    function register(bytes32 did, address key, bytes32 vcHash, bytes32 certHash) external onlyCa {
        require(identities[did].registeredAt == 0, "exists");
        identities[did] = Identity(key, vcHash, certHash, uint64(block.timestamp), 0);
        emit Registered(did, key, vcHash);
    }

    function revoke(bytes32 did) external {
        require(msg.sender == ca || msg.sender == governance, "auth");
        Identity storage it = identities[did];
        require(it.registeredAt != 0 && it.revokedAt == 0, "state");
        it.revokedAt = uint64(block.timestamp);
        emit Revoked(did, it.revokedAt);
    }

    function isValid(bytes32 did) public view returns (bool) {
        Identity storage it = identities[did];
        return it.registeredAt != 0 && it.revokedAt == 0;
    }

    function keyOf(bytes32 did) external view returns (address) {
        return identities[did].key;
    }

    function revokedAt(bytes32 did) external view returns (uint64) {
        return identities[did].revokedAt;
    }

    function rotateCa(address newCa, bytes32 newCaKeyHash) external onlyGovernance {
        ca = newCa;
        caKeyHash = newCaKeyHash;
        emit CaRotated(newCa, newCaKeyHash);
    }

    function announceHeKey(bytes32 keyHash, uint64 epoch, uint64 graceEpochs) external onlyGovernance {
        previousHeKeyHash = heKeyHash;
        heKeyHash = keyHash;
        heKeyEpoch = epoch;
        heGraceUntilEpoch = epoch + graceEpochs;
        emit HeKeyAnnounced(keyHash, epoch, heGraceUntilEpoch);
    }

    function acceptsHeKey(bytes32 keyHash, uint64 epoch) external view returns (bool) {
        if (keyHash == heKeyHash) return true;
        return keyHash == previousHeKeyHash && epoch <= heGraceUntilEpoch;
    }
}
