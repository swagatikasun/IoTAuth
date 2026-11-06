// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

library Groth16 {
    uint256 internal constant Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    uint256 internal constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    struct VerifyingKey {
        uint256[2] alpha;
        uint256[4] beta;
        uint256[4] gamma;
        uint256[4] delta;
        uint256[2][] ic;
    }

    struct Proof {
        uint256[2] a;
        uint256[4] b;
        uint256[2] c;
    }

    function g2Generator() internal pure returns (uint256[4] memory) {
        return [
            11559732032986387107991004021392285783925812861821192530917403151452391805634,
            10857046999023057135944570762232829481370756359578518086990519993285655852781,
            4082367875863433681332203403145435568316851327593401208105741076214120093531,
            8495653923123431417604973247489272438418190587263600148770280649306958101930
        ];
    }

    function ecAdd(uint256[2] memory p, uint256[2] memory q) internal view returns (uint256[2] memory r) {
        uint256[4] memory input = [p[0], p[1], q[0], q[1]];
        bool ok;
        assembly {
            ok := staticcall(gas(), 0x06, input, 0x80, r, 0x40)
        }
        require(ok, "ecadd");
    }

    function ecMul(uint256[2] memory p, uint256 s) internal view returns (uint256[2] memory r) {
        uint256[3] memory input = [p[0], p[1], s];
        bool ok;
        assembly {
            ok := staticcall(gas(), 0x07, input, 0x60, r, 0x40)
        }
        require(ok, "ecmul");
    }

    function negate(uint256[2] memory p) internal pure returns (uint256[2] memory) {
        if (p[0] == 0 && p[1] == 0) return p;
        return [p[0], Q - (p[1] % Q)];
    }

    function verify(VerifyingKey storage vk, Proof memory proof, uint256[] memory inputs) internal view returns (bool) {
        require(inputs.length + 1 == vk.ic.length, "inputs");
        uint256[2] memory acc = vk.ic[0];
        for (uint256 i = 0; i < inputs.length; i++) {
            require(inputs[i] < R, "field");
            acc = ecAdd(acc, ecMul(vk.ic[i + 1], inputs[i]));
        }
        uint256[2] memory na = negate(proof.a);
        uint256[24] memory p = [
            na[0], na[1], proof.b[0], proof.b[1], proof.b[2], proof.b[3],
            vk.alpha[0], vk.alpha[1], vk.beta[0], vk.beta[1], vk.beta[2], vk.beta[3],
            acc[0], acc[1], vk.gamma[0], vk.gamma[1], vk.gamma[2], vk.gamma[3],
            proof.c[0], proof.c[1], vk.delta[0], vk.delta[1], vk.delta[2], vk.delta[3]
        ];
        uint256[1] memory out;
        bool ok;
        assembly {
            ok := staticcall(gas(), 0x08, p, 0x300, out, 0x20)
        }
        return ok && out[0] == 1;
    }
}
