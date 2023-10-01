
import { Neuron } from "./neuron.mjs";

export class Layer {
  constructor (neurons) {
    this.neurons = neurons;
    // TODO validate, is array, neuron input counts all match
    this.outputs = Array(this.neurons.length).fill(0);
  }

  model (inputs) {
    this.neurons.forEach((n, i) => {
      this.outputs[i] = n.model(inputs);
    });
//    console.log('I', inputs, 'O', this.outputs)
    return this.outputs.slice();
  }
}