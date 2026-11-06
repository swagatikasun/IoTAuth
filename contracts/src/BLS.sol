// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

library BLS {
    address internal constant G1ADD = address(0x0b);
    address internal constant G1MSM = address(0x0c);
    address internal constant G2ADD = address(0x0d);
    address internal constant PAIRING = address(0x0f);
    address internal constant MAP_FP2_TO_G2 = address(0x11);

    uint256 internal constant P_HI = 0x1a0111ea397fe69a4b1ba7b6434bacd7;
    uint256 internal constant P_LO = 0x64774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab;

    function g1Generator() internal pure returns (bytes memory) {
        return abi.encodePacked(
            bytes16(0), bytes16(0x17f1d3a73197d7942695638c4fa9ac0f), bytes32(0xc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb),
            bytes16(0), bytes16(0x08b3f481e3aaa0f1a09e30ed741d8ae4), bytes32(0xfcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1)
        );
    }

    function negG1Generator() internal pure returns (bytes memory) {
        uint256 yHi = 0x08b3f481e3aaa0f1a09e30ed741d8ae4;
        uint256 yLo = 0xfcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1;
        uint256 lo;
        uint256 borrow;
        unchecked {
            lo = P_LO - yLo;
            borrow = P_LO < yLo ? 1 : 0;
        }
        uint256 hi = P_HI - yHi - borrow;
        return abi.encodePacked(
            bytes16(0), bytes16(0x17f1d3a73197d7942695638c4fa9ac0f), bytes32(0xc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb),
            bytes16(0), bytes16(uint128(hi)), bytes32(lo)
        );
    }

    function fieldElement(bytes32 seed, uint8 tag) internal pure returns (bytes memory) {
        bytes32 a = keccak256(abi.encodePacked(seed, tag, uint8(0)));
        bytes32 b = keccak256(abi.encodePacked(seed, tag, uint8(1)));
        uint128 top = uint128(uint256(a) >> 128) & 0x0fffffffffffffffffffffffffffffff;
        return abi.encodePacked(bytes16(0), bytes16(top), b);
    }

    function hashToG2(bytes32 message) internal view returns (bytes memory out) {
        bytes memory input = abi.encodePacked(fieldElement(message, 0), fieldElement(message, 1));
        bool ok;
        (ok, out) = MAP_FP2_TO_G2.staticcall(input);
        require(ok && out.length == 256, "map");
    }

    function addG1(bytes memory a, bytes memory b) internal view returns (bytes memory out) {
        bool ok;
        (ok, out) = G1ADD.staticcall(abi.encodePacked(a, b));
        require(ok && out.length == 128, "g1add");
    }

    function addG2(bytes memory a, bytes memory b) internal view returns (bytes memory out) {
        bool ok;
        (ok, out) = G2ADD.staticcall(abi.encodePacked(a, b));
        require(ok && out.length == 256, "g2add");
    }

    function mulG1(bytes memory point, uint256 scalar) internal view returns (bytes memory out) {
        bool ok;
        (ok, out) = G1MSM.staticcall(abi.encodePacked(point, scalar));
        require(ok && out.length == 128, "g1msm");
    }

    function verify(bytes memory pk, bytes32 message, bytes memory signature) internal view returns (bool) {
        bytes memory h = hashToG2(message);
        bytes memory input = abi.encodePacked(pk, h, negG1Generator(), signature);
        (bool ok, bytes memory out) = PAIRING.staticcall(input);
        return ok && out.length == 32 && abi.decode(out, (uint256)) == 1;
    }

    function aggregate(bytes[] memory points) internal view returns (bytes memory acc) {
        acc = points[0];
        for (uint256 i = 1; i < points.length; i++) {
            acc = addG1(acc, points[i]);
        }
    }
}
