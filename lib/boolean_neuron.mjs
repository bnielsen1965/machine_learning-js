
import { Neuron } from "./neuron.mjs";

export class BooleanNeuron extends Neuron {
  constructor (inputWeights, bias, activation) {
    super (inputWeights, bias, activation || BooleanNeuron.activation);
  }

  // use round to produce binary values
  static activation (x) {
    return Math.round(BooleanNeuron.sigmoid(x));
  }
  
  // squashes x value into the range of 0 to 1 where x = 0 returns 0.5
  static sigmoid (x) {
    return 1 / (1 + Math.exp(-1 * x));
  }
}
