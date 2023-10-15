

export class NeuronNetwork {
  constructor (layers) {
    this.layers = layers;
  }

  model (inputs) {
    let outputs = inputs.slice();
    this.layers.forEach(layer => outputs = layer.model(outputs));
    return outputs;
  }
}