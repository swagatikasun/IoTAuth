// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BLS} from "../BLS.sol";

contract BLSSigner {
    address internal constant G2MSM = address(0x0e);

    function publicKey(uint256 sk) external view returns (bytes memory) {
        return BLS.mulG1(BLS.g1Generator(), sk);
    }

    function sign(uint256 sk, bytes32 message) public view returns (bytes memory out) {
        bytes memory h = BLS.hashToG2(message);
        bool ok;
        (ok, out) = G2MSM.staticcall(abi.encodePacked(h, sk));
        require(ok && out.length == 256, "g2msm");
    }

    function aggregateSign(uint256[] calldata sks, bytes32 message) external view returns (bytes memory acc) {
        acc = sign(sks[0], message);
        for (uint256 i = 1; i < sks.length; i++) acc = BLS.addG2(acc, sign(sks[i], message));
    }
}
