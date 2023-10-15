

import { BooleanNeuron } from "./lib/boolean_neuron.mjs";
import { BooleanTraining } from "./lib/boolean_training.mjs";
import { NeuronLayer } from "./lib/neuron_layer.mjs";
import { NeuronNetwork } from "./lib/neuron_network.mjs";

// data definition of XOR gate
const DataXOR = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 1 },
  { inputs: [1, 0], result: 1 },
  { inputs: [1, 1], result: 0 }
];




// Target acceptable error rate for neuron model
const ErrorTarget = 0.0001;

// Maximum number of training runs to try before failure
const MaxRuns = 400000;

// learning rate applied to input weight to adjust for error
const LearnRate = 0.1;

// bias and error rate tweaks applied during training
const Tweak = 0.001;



// first layer with two neurons
let layer1 = new NeuronLayer([
  new BooleanNeuron([Math.random(), Math.random()], Math.random()),
  new BooleanNeuron([Math.random(), Math.random()], Math.random())
]);

// second layer with single neuron for final output
let layer2 = new NeuronLayer([
  new BooleanNeuron([Math.random(), Math.random()], Math.random())
]);

// create network from neuron layers
let network = new NeuronNetwork([layer1, layer2]);



// test xor function
console.log("Untrained XOR output:");
BooleanTraining.showOutput(network, DataXOR);



// first training layer with two neurons
let trainingLayer1 = new NeuronLayer([
  new BooleanNeuron(layer1.neurons[0].weights, layer1.neurons[0].bias, BooleanNeuron.sigmoid),
  new BooleanNeuron(layer1.neurons[1].weights, layer1.neurons[1].bias, BooleanNeuron.sigmoid)
]);

// second training layer with single neuron for final output
let trainingLayer2 = new NeuronLayer([
  new BooleanNeuron(layer2.neurons[0].weights, layer2.neurons[0].bias, BooleanNeuron.sigmoid)
]);

// create training network from neuron layers
let trainingNetwork = new NeuronNetwork([trainingLayer1, trainingLayer2]);

console.log("\nTrain network...");
console.log(`Initial network error ${BooleanTraining.calculateError(trainingNetwork, DataXOR)}.`);
let result = trainNetwork(trainingNetwork, DataXOR, LearnRate, Tweak, MaxRuns, ErrorTarget);
console.log(`Trained network error ${result.error} in ${result.count} training runs.\n`);


// copy trained model values to the untrained neuron
layer1.neurons[0].weights = trainingLayer1.neurons[0].weights.slice();
layer1.neurons[0].bias = trainingLayer1.neurons[0].bias;
layer1.neurons[1].weights = trainingLayer1.neurons[1].weights.slice();
layer1.neurons[1].bias = trainingLayer1.neurons[1].bias;

layer2.neurons[0].weights = trainingLayer2.neurons[0].weights.slice();
layer2.neurons[0].bias = trainingLayer2.neurons[0].bias;

BooleanTraining.showOutput(network, DataXOR);

  /*
  * train neuron using data set
  * @param {Object} neuron The neuron object to train.
  * @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
  * @param {Number} rate The learn rate used to adjust the weight on an input.
  * @param {Number} tweak Adjustment applied to the error.
  * @param {Number} runs The maximum number of training runs to attempt.
  * @param {Number} target The target error value for the model.
  */
  function trainNetwork (network, dataSet, rate, tweak, runs, target) {
    let error;
    let count = 0;
    while (++count < runs) {
      // get current model error
      error = BooleanTraining.calculateError(network, dataSet);
      if (isNaN(error)) throw new Error(`Error is NaN!`);
      // check if error target has been achieved
      if (error < target) break;
      // train each network layer
      networkLoop:
      for (const layer of network.layers) {
        for (const neuron of layer.neurons) {

      // get current model error
      error = BooleanTraining.calculateError(network, dataSet);
      if (isNaN(error)) throw new Error(`Error is NaN!`);
      // check if error target has been achieved
      if (error < target) break networkLoop;

          // train each input weight
          let trainWeights = neuron.weights.slice();
          neuron.weights.forEach((weight, i) => {
            let weightRecall = weight;
            neuron.weights[i] += tweak;
            let weightError = BooleanTraining.calculateError(network, dataSet);
            trainWeights[i] = weightRecall - rate * ((weightError - error) / tweak);
            neuron.weights[i] = weightRecall;
          });
          // train neuron bias
          let trainBias = neuron.bias;
          let biasRecall = neuron.bias;
          neuron.bias += tweak;
          let biasError = BooleanTraining.calculateError(network, dataSet);
          trainBias = biasRecall - rate * ((biasError - error) / tweak);
          neuron.bias = biasRecall;

          // apply training results
          neuron.weights = trainWeights.slice();
          neuron.bias = trainBias;

        }
      }

      // validate error reduced
      let adjustError = BooleanTraining.calculateError(network, dataSet);
      if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)
      error = adjustError;
    }
    return { error, count };
  }
