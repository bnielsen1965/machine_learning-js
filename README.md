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

A boolean AND gate takes a number of boolean inputs and outputs a boolean true (1) if all the gate 
inputs are true (1). If any of the gate inputs are false (0) then the output is also false (0).


| input 1 | input 2 | output
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |


