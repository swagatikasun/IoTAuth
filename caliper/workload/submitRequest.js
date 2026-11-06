'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
const { ethers } = require('ethers');

class SubmitRequestWorkload extends WorkloadModuleBase {
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
        this.devices = roundArguments.devices[workerIndex];
        this.counter = 0;
    }

    async submitTransaction() {
        const device = this.devices[this.counter % this.devices.length];
        const wallet = new ethers.Wallet(device.key);
        const nonce = ethers.id(`${this.workerIndex}-${this.roundIndex}-${this.counter}-${Date.now()}`);
        const ciphertextHash = ethers.id(`ct-${this.counter}`);
        const timestamp = Math.floor(Date.now() / 1000);
        const digest = ethers.solidityPackedKeccak256(
            ['string', 'uint256', 'address', 'bytes32', 'bytes32', 'bytes32', 'uint64'],
            ['IoTAuth/request', this.roundArguments.chainId, this.roundArguments.authreq, device.did, ciphertextHash, nonce, timestamp]
        );
        const sig = ethers.Signature.from(await wallet.signMessage(ethers.getBytes(digest)));
        this.counter++;
        return this.sutAdapter.sendRequests({
            contract: 'SCauthreq',
            verb: 'submitRequest',
            args: [device.did, ciphertextHash, nonce, timestamp, sig.v, sig.r, sig.s],
            readOnly: false
        });
    }
}

function createWorkloadModule() {
    return new SubmitRequestWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
