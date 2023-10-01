# Machine Learning in Javascript

This project is a group of experiments in Machine Learning with Javascript.

After watching [Tsoding Daily's - Machine Learning in C](https://www.youtube.com/watch?v=PGSba51aRYU)
I created this project to apply the concepts in Javascript using NodeJS.


# Machine Learning

In software development the a problem is solved by modeling the solution in programming logic. In 
Machine Learning the model is learned by training programmed nerons with a data set and incremental 
neuron adjustments until the neuron based model can repeatedly generate correct solutions within an 
acceptable error margin.


# Neurons

A single neuron has multiple inputs and a single output. The neuron calculates the output value by 
starting with a bias, summing the input values after multiplying by a weight that is given to each 
input, and finally applying an activation method before presenting the neuron output.

```
                              (neuron)
                 /-------------------------------------------\
                /                                             \
                |               \                             |
                |     bias       \                            |
                |       +         \                           |
(input 1) --->  |  (* weight 1)    >  ---> (activation) f(x)  | ---> (output)
                |       +         /                           |
(input 2) --->  |  (* weight 2)  /                            |
                |               /                             |
                |                                             |
                \                                             /
                 \-------------------------------------------/
```

This simple neuron design is easily implemented in a Javascript module...

*neuron.mjs*
```javascript
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
```


# Boolean Experiments

In the experiments neurons will be trained how to perform boolean logic. Boolean logic is both simple to 
understand and foundational to computers and electronics. A boolean holds a binary state, true or false, 
1 or 0, etc. and are used with logic gates to implement solutions. Some of the boolean logic gates 
include AND, OR, and NOR.

A boolean neuron class that extends the general neuron class is provided with an activation function to 
support binary results. The activation is made with a sigmoid function that will squash the neuron's 
calculation results into a floating point value from 0 to 1. The activation function then uses the 
Math round method to convert the floating point value into a binary value.

*boolean_neuron.mjs*
```javascript
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
```

This boolean neuron can be trained to produce gate functions for various boolean gate types.


## AND Gate Experiment

Execute the and.js script in NodeJS to try this experiment.

> node and.js

A boolean AND gate takes a number of boolean inputs and outputs a boolean true (1) if all the gate 
inputs are true (1). If any of the gate inputs are false (0) then the output is also false (0).


| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |


The *and.js* script uses the AND gate data for a two input AND gate to train a boolean neuron to 
function as an AND gate. When the boolean neuron is first created it is initialized with random 
weights on the two inputs and a random bias. An activation function is not specified here as it 
will be automatically assigned in the boolean neuron class constructor. When this new neuron is 
used to show it's output results before training it will most likely have output failures due to 
the currently untrained nature (unless the random results are very lucky).

```javascript
import { BooleanNeuron } from "./lib/boolean_neuron.mjs";

// data set for boolean AND gate
const DataAND = [
  { inputs: [0, 0], result: 0 },
  { inputs: [0, 1], result: 0 },
  { inputs: [1, 0], result: 0 },
  { inputs: [1, 1], result: 1 }
];

// create an untrained random neuron to use as an AND gate
let andGate = new BooleanNeuron([Math.random(), Math.random()], Math.random());

// show random neuron results
console.log("Untrained neuron output...");
show(andGate, DataAND);
```

For the training step a temporariy training boolean neuron is created using the weights and bias 
values from the initial untrained neuron, but in this case the activation function is specified 
as the sigmoid function from the boolean neuron class. For the training step we need to avoid 
converting the neuron output to a binary value to we can calculate the error rate of the neuron's 
current weight and bias model.

The training function is provided with some conditions to assist in the training process. An 
error target is used to tell the training function when the accuracy is close enough to stop 
the training process. A maximum run count is used to prevent an infinite loop if the training 
fails to achieve the target error rate. And the learn rate and tweak values are used to control 
the rate of change applied to the input weights and bias. When training a neuron these values 
can be adjusted as needed to get the desired results within a reasonable time.

```javascript
// create a duplicate training neuron with sigmoid activation so we get linear error results
let train_neuron = new BooleanNeuron(andGate.weights, andGate.bias, BooleanNeuron.sigmoid);

console.log("\nTrain neuron...");
let error_target = 0.0001;
let max_runs = 400000;
let learn_rate = 0.1;
let tweak = 0.001;

console.log(`Initial neuron error ${calculateError(train_neuron, DataAND)}.`);
let result = train(train_neuron, DataAND, learn_rate, tweak, max_runs, error_target);
console.log(`Trained neuron error ${result.error} in ${result.count} training runs.\n`);

// copy trained model values to the untrained neuron
andGate.weights = train_neuron.weights.slice();
andGate.bias = train_neuron.bias;

if (!test(andGate, DataAND)) console.log("Training failed, neuron does not produce expected output.");
console.log("Trained neuron output...");
show(andGate, DataAND);
```

Training of the neuron is dependant on knowing how far off the neuron's model is from producing 
the correct results for the assigned problem. The error calculation function takes the current 
neuron model and compares the output to the training data set to determine the level of error in 
the current neuron model.

```javascript
/*
* quantify the error in the neuron model compared to training data set
* @param {Object} neuron The neuron object to train.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
*/
function calculateError (neuron, dataSet) {
  let error_val = 0;
  for (const row of dataSet) {
    let y = neuron.model(row.inputs);
    error_val += Math.pow(y - row.result, 2);
  }
  return error_val / dataSet.length;
}
```

The training method iterates through multiple adjustments to the neuron's model until the error 
calculation produces an acceptable error rate or the training steps reaches the maximum allowed 
iterations.

```javascript

/*
* train neuron using data set
* @param {Object} neuron The neuron object to train.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
* @param {Number} rate The learn rate used to adjust the weight on an input.
* @param {Number} tweak Adjustment applied to the error.
* @param {Number} runs The maximum number of training runs to attempt.
* @param {Number} target The target error value for the model.
*/
function train (neuron, dataSet, rate, tweak, runs, target) {
  let error;
  let count = 0;
  while (++count < runs) {
    // get current model error
    error = calculateError(neuron, dataSet);
    if (isNaN(error)) throw new Error(`Error is NaN!`);
    // check if error target has been achieved
    if (error < target) break;
    // train each input weight
    let trainWeights = neuron.weights.slice();
    neuron.weights.forEach((weight, i) => {
      let weightRecall = weight;
      neuron.weights[i] += tweak;
      let weightError = calculateError(neuron, dataSet);
      trainWeights[i] = weightRecall - rate * ((weightError - error) / tweak);
      neuron.weights[i] = weightRecall;
    });
    // train neuron bias
    let trainBias = neuron.bias;
    let biasRecall = neuron.bias;
    neuron.bias += tweak;
    let biasError = calculateError(neuron, dataSet);
    trainBias = biasRecall - rate * ((biasError - error) / tweak);
    neuron.bias = biasRecall;

    // apply training results
    neuron.weights = trainWeights.slice();
    neuron.bias = trainBias;

    // validate error reduced
    let adjustError = calculateError(neuron, dataSet);
    if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)
    error = adjustError;
  }
  return { error, count };
}
```