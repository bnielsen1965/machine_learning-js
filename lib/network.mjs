
import { Layer } from "./layer.mjs";

export class Network {
  constructor (layers) {
    this.layers = layers;
    // TODO validation, is array, input/output counts align from layer to layer
  }

  model (inputs) {
    let outputs = inputs.slice();
    this.layers.forEach(l => {
      outputs = l.model(outputs);
    });
    return outputs;
  }

  // train using data set
  train (dataSet, runs, rate, tweak) {
    while (runs--) {
      // get current model error
      let error = this.calculateError(dataSet);
      console.log('E', error);
      if (isNaN(error)) throw new Error(`Error is NaN!`);
      // train each layer
      this.layers.forEach((layer, li) => {
        // train each neuron
        layer.neurons.forEach((neuron, ni) => {
          // train each input weight
          neuron.weights.forEach((weight, i) => {
            let weightRecall = weight;
            neuron.weights[i] += tweak;
            let tweakError = this.calculateError(dataSet);
            neuron.trainWeights[i] = weightRecall - rate * ((tweakError - error) / tweak);
            neuron.weights[i] = weightRecall;
          });
          // train neuron bias
          let biasRecall = neuron.bias;
          neuron.bias += tweak;
          let biasError = this.calculateError(dataSet);
          neuron.trainBias = biasRecall - rate * ((biasError - error) / tweak);
          neuron.bias = biasRecall;
        });
      });

      // apply training results
      this.layers.forEach(layer => {
        // train each neuron
        layer.neurons.forEach(neuron => {
          // train each input weight
          neuron.weights.forEach((weight, i) => {
            neuron.weights[i] = neuron.trainWeights[i];
          });
          neuron.bias = neuron.trainBias;
        });
      });

      // validate error reduced
      let adjustError = this.calculateError(dataSet);
      if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)


      // TODO need to loop through and tweak every neuron input weights and bias and adjust each

      // need to reach into layers and temporarily adjust neuron weight/bias to produce error then reset

      // // get error for current weights
      // let weights = this.weights.slice();
      // let bias = this.bias;
      // let error = this.trainError(dataSet, weights, bias);
      // // tweak each input weight and adjust weight based on error
      // for (let i = 0; i < weights.length; i++) {
      //   let trainWeights = weights.slice();
      //   trainWeights[i] += tweak;
      //   let tweakError = this.trainError(dataSet, trainWeights, bias);
      //   this.weights[i] -= rate * ((tweakError - error) / tweak);
      // }
      // // tweak bias
      // let biasError = this.trainError(dataSet, weights, bias + tweak);
      // this.bias -= rate * ((biasError - error) / tweak);
    }
  }

  // quantify the error in the model compared to training data set
  calculateError (dataSet) {
    let error_val = 0;
    for (const row of dataSet) {
      let y = this.model(row.inputs);
      error_val += Math.pow(y[0] - row.result, 2);
    }
    return error_val / dataSet.length;
  }
}