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

This simple neuron design is easily implemented in the *neuron.mjs* Javascript module...

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
```

This boolean neuron can be trained to produce gate functions for various boolean gate types.


## Gate learning experiment in gates.js

Execute the gates.js script in NodeJS to try this experiment.

> node gates.js

The gates.js script has definitions for two input AND, NAND, OR, and NOR logic gates. The 
logic for each gate are as follows...

AND gate:

| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

NAND gate:

| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

OR gate:

| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

NOR gate:

| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |


These gate definition tables are used to train individual boolean neurons to act as a gate for each 
of the defined types. When each boolean neuron is first created it is initialized with random 
weights on the two inputs and a random bias. The untrained neuron is tested against the logic 
gate data set and the result will more than likely fail.


Example output for an untrained neuron...
```
Training AND...
Untrained neuron output:
In: [ 0 | 0 ] Out: 1,  expected 0 FAILURE
In: [ 0 | 1 ] Out: 1,  expected 0 FAILURE
In: [ 1 | 0 ] Out: 1,  expected 0 FAILURE
In: [ 1 | 1 ] Out: 1,  expected 1 SUCCESS

```

For the training step a temporariy training boolean neuron is created using the weights and bias 
values from the random untrained neuron. In the training boolean neuron the boolean activation 
function is overridden with the sigmoid function which returns a floating point value between 0 
and 1 thus enabling a measure of the error between the expected output versus the actual output 
of the neuron.

The training function is provided with some conditions to assist in the training process. An 
error target is used to tell the training function when the accuracy is close enough to stop 
the training process. A maximum run count is used to prevent an infinite loop if the training 
fails to achieve the target error rate. And the learn rate and tweak values are used to control 
the rate of change applied to the input weights and bias. When training a neuron these values 
can be adjusted as needed to get the desired results within a reasonable time.

```javascript
// Target acceptable error rate for neuron model
const ErrorTarget = 0.0001;

// Maximum number of training runs to try before failure
const MaxRuns = 400000;

// learning rate applied to input weight to adjust for error
const LearnRate = 0.1;

// bias and error rate tweaks applied during training
const Tweak = 0.001;
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
  static calculateError (neuron, dataSet) {
    let error_val = 0;
    for (const row of dataSet) {
      let y = neuron.model(row.inputs);
      error_val += Math.pow(y - row.result, 2);
    }
    return error_val / dataSet.length;
  }
```

If the error level in the training neuron is not at an acceptable level then the weights and bias
of the training neuron are adjusted to reduce the error level. These adjustments are calculated 
for each weight and the bias of the training neuron.

```javascript
/*
* adjust neuron weights and bias to reduce the error
* @param {Object} neuron The neuron object to adjust.
* @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
* @param {Number} rate The learn rate used to adjust the weight on an input.
* @param {Number} tweak Adjustment applied to the error.
* @param {Number} error The current error value for the neuron/network.
* @param {Object} network (optional) The network object if adjusting a neuron inside a network.
*/
static adjustWeightsBias (neuron, dataSet, rate, tweak, error, network) {
  let trainWeights = neuron.weights.slice();
  neuron.weights.forEach((weight, i) => {
    let weightRecall = weight;
    neuron.weights[i] += tweak;
    // calculate neuron error
    let weightError = BooleanTraining.calculateError(network || neuron, dataSet);
    trainWeights[i] = weightRecall - rate * ((weightError - error) / tweak);
    neuron.weights[i] = weightRecall;
  });
  // train neuron bias
  let trainBias = neuron.bias;
  let biasRecall = neuron.bias;
  neuron.bias += tweak;
  // calculate neuron error
  let biasError = BooleanTraining.calculateError(network || neuron, dataSet);
  trainBias = biasRecall - rate * ((biasError - error) / tweak);
  neuron.bias = biasRecall;

  // apply training results
  neuron.weights = trainWeights.slice();
  neuron.bias = trainBias;
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
      BooleanTraining.adjustWeightsBias(neuron, dataSet, rate, tweak, error);
      // validate error reduced
      let adjustError = BooleanTraining.calculateError(neuron, dataSet);
      if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)
      error = adjustError;
    }
    return { error, count };
  }
```

As each neuron is trained to operate as a logic gate the final results of the training will 
be displayed and should show success when tested against the gate definition.

```
Training AND...

Trained neuron output...
In: [ 0 | 0 ] Out: 0,  expected 0 SUCCESS
In: [ 0 | 1 ] Out: 0,  expected 0 SUCCESS
In: [ 1 | 0 ] Out: 0,  expected 0 SUCCESS
In: [ 1 | 1 ] Out: 1,  expected 1 SUCCESS
Neuron Model:
  bias: -13.202136067669324
  weights:
    input 0: 8.743925936900641
    input 1: 8.743925936900641

```

The results of the training will be stored in a JSON file named gate_models.json 
which can then be used in other scripts to create trained boolean neurons for 
use in other functions.


## XOR gate from 3 gates in 3gate-xor.js

In an exclusive OR gate the output is 1 (true) only when one of the inputs is 1 (true).
If you attempt to train a single boolean neuron to act as an XOR logic gate the 
training will always fail as a single neuron is not capable of representing an XOR gate.

XOR gate:

| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

In boolean algebra, Karnaugh maps, and electronics design, it is common to combine logic gates 
to create a new logic gate that matches the defined logic. For an XOR gate we can 
combine a NAND gate, OR gate, and AND gate to create the desired logic.

The two data inputs are supplied to the NAND and OR gates then the output of these two 
gates is fed into the AND gate inputs. The final output of the AND gate will be the 
exclusive OR of the A and B inputs.

In code it looks something like this...
```javascript
let nandGate = new BooleanNeuron(nandModel.weights, nandModel.bias);
let orGate = new BooleanNeuron(orModel.weights, orModel.bias);
let andGate = new BooleanNeuron(andModel.weights, andModel.bias);

// xor function from 3 gates
function xor (a, b) {
  let nandOut = nandGate.model([a, b]);
  let orOUt = orGate.model([a, b]);
  let andOut = andGate.model([nandOut, orOUt]);
  return andOut;
}
```

The function xor() takes the two inputs and uses the three boolean neuron gates to 
produce an XOR output.

```
XOR output based on 3 gate OR, NAND, AND design:
[ 0 | 0 ] == 0
[ 0 | 1 ] == 1
[ 1 | 0 ] == 1
[ 1 | 1 ] == 0

```


## XOR gate neural network

An alternative to a multi-gate solution where multiple neurons, each trained to 
act as a specific type of logic gate, we can in its place train a network of neurons to 
perform the function of an XOR gate.

The neural network wiring is similar to the three gate design where two inputs are 
connected to the inputs of two neurons in the first layer and the output of those two 
neurons are then connected to the inputs of a single neuron in the final layer. The 
output of the neuron in the final layer is the gate output.

Two new classes are created to associate the neurons in a neural network. The NeuronLayer 
class which contains all the neurons in a network layer. And a NeuronNetwork class 
that contains all the layers for the network.

In this experiment the first layer contains an array of two random neurons with two inputs. 
The second layer contains an array with a single random neuron, this will be the final 
stage for output. And the neural network is constructed with an array of the layers.
```javascript

// first layer with two neurons
let layer1 = new NeuronLayer([
  new BooleanNeuron([Math.random(), Math.random()], Math.random()),
  new BooleanNeuron([Math.random(), Math.random()], Math.random())
]);

// second layer with single neuron for final output
let layer2 = new NeuronLayer([
  new BooleanNeuron([Math.random(), Math.random()], Math.random())
]);

// create network from neuron layers
let network = new NeuronNetwork([layer1, layer2]);
```

When the XOR data set is ran through this network of random neurons it will more than 
likely fail the tests.
```
Untrained XOR output:
In: [ 0 | 0 ] Out: 1,  expected 0 FAILURE
In: [ 0 | 1 ] Out: 1,  expected 1 SUCCESS
In: [ 1 | 0 ] Out: 1,  expected 1 SUCCESS
In: [ 1 | 1 ] Out: 1,  expected 0 FAILURE

```

The training is similar to that of the single neurons in the gates experiment. A 
training network is created to duplicate the model of the untrained XOR network with 
the activation functions swapped out for a sigmoid method to enable a measure of the 
error rate in the current model.

```javascript
// first training layer with two neurons
let trainingLayer1 = new NeuronLayer([
  new BooleanNeuron(layer1.neurons[0].weights, layer1.neurons[0].bias, BooleanNeuron.sigmoid),
  new BooleanNeuron(layer1.neurons[1].weights, layer1.neurons[1].bias, BooleanNeuron.sigmoid)
]);

// second training layer with single neuron for final output
let trainingLayer2 = new NeuronLayer([
  new BooleanNeuron(layer2.neurons[0].weights, layer2.neurons[0].bias, BooleanNeuron.sigmoid)
]);

// create training network from neuron layers
let trainingNetwork = new NeuronNetwork([trainingLayer1, trainingLayer2]);
```

The training neural network is then ran through a network traning method with the 
XOR data set until the training has reached an accetable error rate or has ran the 
maximum number of training runs. The network training method is similar to the training 
method for a single neuron except the network method must interate through all layers 
and all neurons.

```javascript
  /*
  * train neuron using data set
  * @param {Object} network The neuron network object to train.
  * @param {Object[]} dataSet An array of objects with the neuron input values and expected output result.
  * @param {Number} rate The learn rate used to adjust the weight on an input.
  * @param {Number} tweak Adjustment applied to the error.
  * @param {Number} runs The maximum number of training runs to attempt.
  * @param {Number} target The target error value for the model.
  */
  static trainNetwork (network, dataSet, rate, tweak, runs, target) {
    let error;
    let count = 0;
    while (++count < runs) {
      // get current model error
      error = BooleanTraining.calculateError(network, dataSet);
      if (isNaN(error)) throw new Error(`Error is NaN!`);
      // check if error target has been achieved
      if (error < target) break;
      // train each network layer
      networkLoop:
      for (const layer of network.layers) {
        for (const neuron of layer.neurons) {
          // get current model error
          error = BooleanTraining.calculateError(network, dataSet);
          if (isNaN(error)) throw new Error(`Error is NaN!`);
          // check if error target has been achieved
          if (error < target) break networkLoop;
          // train each input weight
          BooleanTraining.adjustWeightsBias(neuron, dataSet, rate, tweak, error, network);
        }
      }

      // validate error reduced
      let adjustError = BooleanTraining.calculateError(network, dataSet);
      if (adjustError > error) console.log(`Adjustment failed! ${adjustError} > ${error}`)
      error = adjustError;
    }
    return { error, count };
  }
```

Once training is complete the weights and bias values of the training model are copied into the 
XOR neural network. Now when the XOR data set is ran through the trained XOR neural network it 
should pass all tests.
```
Trained neuron output...
In: [ 0 | 0 ] Out: 0,  expected 0 SUCCESS
In: [ 0 | 1 ] Out: 1,  expected 1 SUCCESS
In: [ 1 | 0 ] Out: 1,  expected 1 SUCCESS
In: [ 1 | 1 ] Out: 0,  expected 0 SUCCESS
```


### XOR neural network vs 3 gate XOR

A comparison of the weights and bias values of the XOR neural network neurons to the trained 
gate neurons from the 3 gate XOR design may provide some interesting results. The models in 
most cases will be very different and the XOR neural network model may produce a different 
model result from each run due to a random starting point, while the single gate models tend 
to learn very similar models even though they also start from a random point.


# Web Browser Neuron Experiments

The javascript code for the neuron experiments can be executed in a web browser as well. The 
html directory in the project includes html files that can be loaded directly in a web browser 
and includes testing and training capabilities as well as a visual display of the neuron 
function.
