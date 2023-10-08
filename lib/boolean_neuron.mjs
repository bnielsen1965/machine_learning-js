
import { Neuron } from "./neuron.mjs";

/*
* The BooleanNeuron class extends the Neuron class by providing a predefined 
* activation function that is used to ensure the neuron model output adheres 
* to boolean values of 0 or 1.
*/

export class BooleanNeuron extends Neuron {
  /*
  * @constructor
  * @param {Number[]} inputWeights An array of initial input weights.
  * @param {Number} bias An initial bias value.
  * @param {function} [activation=BooleanNeuron.activation] The activation function 
  *     can be overridden when creating a training neuron that requires a non-sigmoid
  *     output for error calculations.
  */
  constructor (inputWeights, bias, activation) {
    super (inputWeights, bias, activation || BooleanNeuron.activation);
  }

  /*
  * The activation function uses round() to produce boolean values from the sigmoid.
  * @param {Number} x The sumation of the bias, input and weight calculations.
  */
  static activation (x) {
    return Math.round(BooleanNeuron.sigmoid(x));
  }
  
  /*
  * A sigmoid function is used to squash x value into the range of 0 to 1 where x = 0 returns 0.5.
  * @param {Number} x The value to be squashed into the range from 0 to 1.
  */
  static sigmoid (x) {
    return 1 / (1 + Math.exp(-1 * x));
  }

  // export the model definition values
  getModel () {
    return {
      bias: this.bias,
      weights: this.weights
    };
  }
}
