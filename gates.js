
/*
* The gates script uses the machine learning boolean neuron and definitions
* for AND, NAND, OR, and NOR gates to learn neuron models to represent the 
* logic gates.
*
* The logic gate model results will be saved in a JSON file for use in other 
* scripts.
*/


import { BooleanNeuron } from "./lib/boolean_neuron.mjs";
import { BooleanTraining } from "./lib/boolean_training.mjs";
import * as FS from "fs";

// definition of 2 input AND gate
const DataAND = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 0 },
  { inputs: [1, 0], result: 0 },
  { inputs: [1, 1], result: 1 }
];

// definition of 2 input NAND gate
const DataNAND = [
  { inputs: [0, 0], result: 1 },
  { inputs: [0, 1], result: 1 },
  { inputs: [1, 0], result: 1 },
  { inputs: [1, 1], result: 0 }
];

// definition of 2 input OR gate
const DataOR = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 1 },
  { inputs: [1, 0], result: 1 },
  { inputs: [1, 1], result: 1 }
];

// definition of 2 input NOR gate
const DataNOR = [
  { inputs: [0, 0], result: 1 },
  { inputs: [0, 1], result: 0 },
  { inputs: [1, 0], result: 0 },
  { inputs: [1, 1], result: 0 }
];

// array of defined training sets for logic gates
const TrainingSets = [
  { name: "AND", data: DataAND },
  { name: "NAND", data: DataNAND },
  { name: "OR", data: DataOR },
  { name: "NOR", data: DataNOR }
];


// Target acceptable error rate for neuron model
const ErrorTarget = 0.0001;

// Maximum number of training runs to try before failure
const MaxRuns = 400000;

// learning rate applied to input weight to adjust for error
const LearnRate = 0.1;

// bias and error rate tweaks applied during training
const Tweak = 0.001;

// learned gate models will be stored in this array
let gateModels = [];

// process each training set
for (const trainingSet of TrainingSets) {
  console.log(`Training ${trainingSet.name}...`);

  // create an untrained random neuron to use as a boolean logic gate
  let gate = new BooleanNeuron([Math.random(), Math.random()], Math.random());

  // show random neuron results
  console.log("Untrained neuron output:");
  BooleanTraining.showOutput(gate, trainingSet.data);
  BooleanTraining.showModel(gate);

  // create a duplicate training neuron with sigmoid activation so we get linear error results
  let train_neuron = new BooleanNeuron(gate.weights, gate.bias, BooleanNeuron.sigmoid);
  
  console.log("\nTrain neuron...");
  console.log(`Initial neuron error ${BooleanTraining.calculateError(train_neuron, trainingSet.data)}.`);
  let result = await BooleanTraining.train(train_neuron, trainingSet.data, LearnRate, Tweak, MaxRuns, ErrorTarget);
  console.log(`Trained neuron error ${result.error} in ${result.count} training runs.\n`);

  // copy trained model values to the untrained neuron
  gate.weights = train_neuron.weights.slice();
  gate.bias = train_neuron.bias;
  
  if (!BooleanTraining.test(gate, trainingSet.data)) console.log("Training failed, neuron does not produce expected output.");
  console.log("Trained neuron output...");
  BooleanTraining.showOutput(gate, trainingSet.data);
  BooleanTraining.showModel(gate);
  console.log("\n\n");

  gateModels.push({ name: trainingSet.name, model: gate.getModel() });
}

// save the trained gate models
FS.writeFileSync("gate_models.json", JSON.stringify(gateModels, null, 2));

