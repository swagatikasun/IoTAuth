// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BLS} from "../src/BLS.sol";

library BLSHelper {
    address internal constant G2MSM = address(0x0e);

    function publicKey(uint256 sk) internal view returns (bytes memory) {
        return BLS.mulG1(BLS.g1Generator(), sk);
    }

    function sign(uint256 sk, bytes32 message) internal view returns (bytes memory out) {
        bytes memory h = BLS.hashToG2(message);
        bool ok;
        (ok, out) = G2MSM.staticcall(abi.encodePacked(h, sk));
        require(ok && out.length == 256, "g2msm");
    }

    function aggregateSignatures(bytes[] memory sigs) internal view returns (bytes memory acc) {
        acc = sigs[0];
        for (uint256 i = 1; i < sigs.length; i++) {
            acc = BLS.addG2(acc, sigs[i]);
        }
    }
}
