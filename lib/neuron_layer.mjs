
/*
* A Layer class instance is used to contain a list of neurons that are all connected to the same inputs
* and output the results of all neurons in the layer.
*
* NOTE: If the layer has multiple neurons then the output is an array of values. If the layer contains 
* a single neuron then the output is the result from the single neuron.
*/

export class NeuronLayer {
  /*
  * @constructor
  * @param {neuron[]} neurons An array of neuron objects.
  */
  constructor (neurons) {
    this.neurons = neurons;
    this.outputs = Array(this.neurons.length).fill(0);
  }

  /*
  * Calculates neuron layer output from given inputs.
  * @param {Number[]} inputs An array of neuron input values.
  */
  model (inputs) {
    // if layer has single neuron then return the neuron's output
    if (this.neurons.length === 1) return this.neurons[0].model(inputs);
    // process inputs through each neuron
    this.neurons.forEach((n, i) => this.outputs[i] = n.model(inputs));
    return this.outputs.slice();
  }
}
