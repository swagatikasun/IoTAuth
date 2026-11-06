// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {BLS} from "../src/BLS.sol";
import {BLSHelper} from "./BLSHelper.sol";

contract BLSTest is Test {
    function testSignVerify() public view {
        bytes memory pk = BLSHelper.publicKey(12345);
        bytes32 m = keccak256("root");
        bytes memory sig = BLSHelper.sign(12345, m);
        assertTrue(BLS.verify(pk, m, sig));
        assertFalse(BLS.verify(pk, keccak256("other"), sig));
    }

    function testAggregate() public view {
        bytes32 m = keccak256("batch");
        bytes[] memory pks = new bytes[](3);
        bytes[] memory sigs = new bytes[](3);
        for (uint256 i = 0; i < 3; i++) {
            pks[i] = BLSHelper.publicKey(1000 + i);
            sigs[i] = BLSHelper.sign(1000 + i, m);
        }
        assertTrue(BLS.verify(BLS.aggregate(pks), m, BLSHelper.aggregateSignatures(sigs)));
    }
}
