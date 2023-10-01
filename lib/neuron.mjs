
export class Neuron {
  /*
  * Represents a single neuron
  * @constructor
  * @param {Number[]} inputWeights An array of initial input weights.
  * @param {Number} bias An initial bias value.
  * @param {function} activation The activation function to apply before output.
  */
  constructor (inputWeights, bias, activation) {
    if (!inputWeights || !inputWeights.length) throw new Error("New neuron requires input weights.");
    if (!Array.isArray(inputWeights)) throw new Error("Input weights must be an array of weights.");
    this.activation = activation;
    this.bias = bias || 0;
    this.weights = inputWeights.slice();
  }

  /*
  * Calculates neuron output from given inputs.
  * @param {Number[]} inputs An array of neuron input values.
  */
  model (inputs) {
    if (inputs.length != this.weights.length) throw new Error("Mismatch between number of data inputs and number of neuron inputs.");
    let result = this.bias;
    for (let i = 0; i < inputs.length; i++) result += inputs[i] * this.weights[i];
    return this.activation ? this.activation(result) : result;
  }
}
