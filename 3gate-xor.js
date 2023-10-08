
/*
* The 3gate-xor script uses the boolean neuron models for NAND, OR, and AND 
* gates to create an xor function constructed from three boolean neurons.
*/

import { BooleanNeuron } from "./lib/boolean_neuron.mjs";
import * as FS from "fs";

// data definition of XOR gate
const DataXOR = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 1 },
  { inputs: [1, 0], result: 1 },
  { inputs: [1, 1], result: 0 }
];

// try reading the gate_models.json file created by the gates.js script
let gateModelsJSON;
try {
  FS.statSync("gate_models.json");
  gateModelsJSON = FS.readFileSync("gate_models.json");
}
catch (error) {
  if (error.code == "ENOENT") {
    console.error(`The gate_models.json file does not exist. You must run the gates.js script to create the models first.`);
    process.exit(1);
  }
  console.error(`Failed reading the gates_models.json file. ${error.message}`);
  process.exit(1);
}

// extract NAND, OR, and AND models and create gates
let gateModels = JSON.parse(gateModelsJSON);
let nandModel = gateModels.filter(model => model.name == "NAND")[0].model;
let orModel = gateModels.filter(model => model.name == "OR")[0].model;
let andModel = gateModels.filter(model => model.name == "AND")[0].model;

let nandGate = new BooleanNeuron(nandModel.weights, nandModel.bias);
let orGate = new BooleanNeuron(orModel.weights, orModel.bias);
let andGate = new BooleanNeuron(andModel.weights, andModel.bias);

// xor function from 3 gates
function xor (a, b) {
  let nandOut = nandGate.model([a, b]);
  let orOUt = orGate.model([a, b]);
  let andOut = andGate.model([nandOut, orOUt]);
  return andOut;
}

// test xor function
console.log("XOR output based on 3 gate OR, NAND, AND design:");
DataXOR.forEach(d => {
  let xorOut = xor(d.inputs[0], d.inputs[1]);
  console.log(`[ ${d.inputs.join(" | ")} ] == ${xorOut}`);
});
