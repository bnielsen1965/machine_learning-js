
import { BooleanNeuron } from "./lib/boolean_neuron.mjs";

// data set for boolean AND gate
const DataAND = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 0 },
  { inputs: [1, 0], result: 0 },
  { inputs: [1, 1], result: 1 }
];

// create an untrained random neuron to use as an AND gate
let andGate = new BooleanNeuron([Math.random(), Math.random()], Math.random());

// show random neuron results
console.log("Untrained neuron output...");
show(andGate, DataAND);

// create a duplicate training neuron with sigmoid activation so we get linear error results
let train_neuron = new BooleanNeuron(andGate.weights, andGate.bias, BooleanNeuron.sigmoid);

console.log("\nTrain neuron...");
let error_target = 0.0001;
let max_runs = 400000;
let learn_rate = 0.1;
let tweak = 0.001;

console.log(`Initial neuron error ${calculateError(train_neuron, DataAND)}.`);
let result = train(train_neuron, DataAND, learn_rate, tweak, max_runs, error_target);
console.log(`Trained neuron error ${result.error} in ${result.count} training runs.\n`);

// copy trained model values to the untrained neuron
andGate.weights = train_neuron.weights.slice();
andGate.bias = train_neuron.bias;

if (!test(andGate, DataAND)) console.log("Training failed, neuron does not produce expected output.");
console.log("Trained neuron output...");
show(andGate, DataAND);


/*
* train neuron using data set
* @param {Object} neuron The neuron object to train.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
* @param {Number} rate The learn rate used to adjust the weight on an input.
* @param {Number} tweak Adjustment applied to the error.
* @param {Number} runs The maximum number of training runs to attempt.
* @param {Number} target The target error value for the model.
*/
function train (neuron, dataSet, rate, tweak, runs, target) {
  let error;
  let count = 0;
  while (++count < runs) {
    // get current model error
    error = calculateError(neuron, dataSet);
    if (isNaN(error)) throw new Error(`Error is NaN!`);
    // check if error target has been achieved
    if (error < target) break;
    // train each input weight
    let trainWeights = neuron.weights.slice();
    neuron.weights.forEach((weight, i) => {
      let weightRecall = weight;
      neuron.weights[i] += tweak;
      let weightError = calculateError(neuron, dataSet);
      trainWeights[i] = weightRecall - rate * ((weightError - error) / tweak);
      neuron.weights[i] = weightRecall;
    });
    // train neuron bias
    let trainBias = neuron.bias;
    let biasRecall = neuron.bias;
    neuron.bias += tweak;
    let biasError = calculateError(neuron, dataSet);
    trainBias = biasRecall - rate * ((biasError - error) / tweak);
    neuron.bias = biasRecall;

    // apply training results
    neuron.weights = trainWeights.slice();
    neuron.bias = trainBias;

    // validate error reduced
    let adjustError = calculateError(neuron, dataSet);
    if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)
    error = adjustError;
  }
  return { error, count };
}

/*
* quantify the error in the neuron model compared to training data set
* @param {Object} neuron The neuron object to train.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
*/
function calculateError (neuron, dataSet) {
  let error_val = 0;
  for (const row of dataSet) {
    let y = neuron.model(row.inputs);
    error_val += Math.pow(y - row.result, 2);
  }
  return error_val / dataSet.length;
}

/*
* test a neuron against a data set
* @param {Object} neuron The neuron object to train.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
*/
function test (neuron, dataSet) {
  let success = true;
  for (const row of dataSet) if (neuron.model(row.inputs) !== row.result) success = false;
  return success;
}

/*
* show neuron output from data set
* @param {Object} neuron The neuron object to train.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
*/
function show (neuron, dataSet) {
  for (const data of dataSet) {
    let output = neuron.model(data.inputs);
    console.log(`Input [${data.inputs[0]}, ${data.inputs[1]}] expects ${data.result}. Neuron output ${output}. ${data.result === output ? "SUCCESS" : "FAILURE"}`);
  }
}
