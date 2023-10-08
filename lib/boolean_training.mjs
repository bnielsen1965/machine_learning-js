
export class BooleanTraining {

  /*
  * train neuron using data set
  * @param {Object} neuron The neuron object to train.
  * @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
  * @param {Number} rate The learn rate used to adjust the weight on an input.
  * @param {Number} tweak Adjustment applied to the error.
  * @param {Number} runs The maximum number of training runs to attempt.
  * @param {Number} target The target error value for the model.
  */
  static train (neuron, dataSet, rate, tweak, runs, target) {
    let error;
    let count = 0;
    while (++count < runs) {
      // get current model error
      error = BooleanTraining.calculateError(neuron, dataSet);
      if (isNaN(error)) throw new Error(`Error is NaN!`);
      // check if error target has been achieved
      if (error < target) break;
      // train each input weight
      let trainWeights = neuron.weights.slice();
      neuron.weights.forEach((weight, i) => {
        let weightRecall = weight;
        neuron.weights[i] += tweak;
        let weightError = BooleanTraining.calculateError(neuron, dataSet);
        trainWeights[i] = weightRecall - rate * ((weightError - error) / tweak);
        neuron.weights[i] = weightRecall;
      });
      // train neuron bias
      let trainBias = neuron.bias;
      let biasRecall = neuron.bias;
      neuron.bias += tweak;
      let biasError = BooleanTraining.calculateError(neuron, dataSet);
      trainBias = biasRecall - rate * ((biasError - error) / tweak);
      neuron.bias = biasRecall;

      // apply training results
      neuron.weights = trainWeights.slice();
      neuron.bias = trainBias;

      // validate error reduced
      let adjustError = BooleanTraining.calculateError(neuron, dataSet);
      if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)
      error = adjustError;
    }
    return { error, count };
  }

  /*
  * quantify the error in the neuron model compared to training data set
  * @param {Object} neuron The neuron object to train.
  * @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
  */
  static calculateError (neuron, dataSet) {
    let error_val = 0;
    for (const row of dataSet) {
      let y = neuron.model(row.inputs);
      error_val += Math.pow(y - row.result, 2);
    }
    return error_val / dataSet.length;
  }

  /*
  * test a neuron against a data set
  * @param {Object} neuron The neuron object to train.
  * @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
  */
  static test (neuron, dataSet) {
    let success = true;
    for (const row of dataSet) if (neuron.model(row.inputs) !== row.result) success = false;
    return success;
  }

  /*
  * show neuron output from data set
  * @param {Object} neuron The neuron object to train.
  * @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
  */
  static showOutput (neuron, dataSet) {
    for (const data of dataSet) {
      let output = neuron.model(data.inputs);
      console.log(`In: [ ${data.inputs.join(" | ")} ] Out: ${output},  expected ${data.result} ${data.result === output ? "SUCCESS" : "FAILURE"}`);
    }
  }

  /*
  * show the boolean model values
  * @param {Object} neuron The neuron object to train.
  */
  static showModel (neuron) {
    let model = neuron.getModel();
    console.log("Neuron Model:");
    console.log(`  bias: ${model.bias}`);
    console.log("  weights:");
    neuron.weights.map((weight, i) => console.log(`    input ${i}: ${weight}`));
  }

}