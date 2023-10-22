

/*
* A NeuronNetwork class instance is used to contain a list of interconnected layers where the inputs of 
* the network are connected to the inputs of the first layer and the outputs of each layer
* are connected to the inputs of the next layer until the final layer that provides the network output.
*
* NOTE: The number of outputs (neurons) in a layer must match the number of inputs in the following layer.
* I.E. Assuming we have two layers where the final layer has one nueron with two inputs, the first 
* layer must have two neurons to provide two layer outputs that will be fed into the two inputs of the
* final layer.
*/

export class NeuronNetwork {
  /*
  * @constructor
  * @param {NeuronLayer[]} layers An array of neuron layer objects.
  */
  constructor (layers) {
    this.layers = layers;
  }

  /*
  * Calculates neuron network output from given inputs.
  * @param {Number[]} inputs An array of neuron input values.
  */
  model (inputs) {
    let outputs = inputs.slice();
    this.layers.forEach(layer => outputs = layer.model(outputs));
    return outputs;
  }
}