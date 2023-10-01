
import { Neuron } from "./lib/neuron.mjs";
import { Layer } from "./lib/layer.mjs";
import { Network } from "./lib/network.mjs";



const dataOr = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 1 },
  { inputs: [1, 0], result: 1 },
  { inputs: [1, 1], result: 1 }
];

const dataAnd = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 0 },
  { inputs: [1, 0], result: 0 },
  { inputs: [1, 1], result: 1 }
];

const dataXor = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 1 },
  { inputs: [1, 0], result: 1 },
  { inputs: [1, 1], result: 0 }
];

const dataLinear = [
  { inputs: [0], result: 0 },
  { inputs: [1], result: 2 },
  { inputs: [2], result: 4 },
  { inputs: [3], result: 6 },
  { inputs: [4], result: 8 }
];


//let d = dataXor;
let train_runs = 400000; //400000;
let learn_rate = 0.1;
let tweak = 0.001;

// let n = new Neuron([Math.random() * 10, Math.random() * 10], Math.random() * 5);
// let l = new Layer([n]);
// let network = new Network([l]);

let data = dataOr;
let neurons = Array.from(Array(3)).map(() => createNeuron());
let layers = [new Layer(neurons.slice(0,2)), new Layer(neurons.slice(2))];
let network = new Network(layers);
showValues(network);
network.train(data, train_runs, learn_rate, tweak);
showValues(network);
for (const row of data) {
  console.log(`given: ${row.inputs} expected ${row.result} produced ${Math.round(network.model(row.inputs))} ${network.model(row.inputs)}`);
//  console.log(`given: ${row.inputs} expected ${row.result} produced ${(network.model(row.inputs))}`);
}




function createNeuron () {
  return new Neuron([Math.random(), Math.random()], Math.random(), activation);
}

function showValues (network) {
  network.layers.forEach((layer, li) => {
    layer.neurons.forEach((neuron, ni) => {
      console.log(li, ni, neuron.bias, neuron.weights)
    })
  })
}


// test model against data set
function test (dataSet) {
  for (const row of dataSet) {
    console.log(`given: ${row.inputs} expected ${row.result} produced ${Math.round(network.model(row.inputs))}`);
  }
}

function activation (result) {
  return 1/(1 + Math.exp(-1 * result));
}