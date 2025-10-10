# Numpy Neural Net

This is primarily a project to demonstrate some foundational concepts of deep learning. A tremendous amount of sophistication is possible in deep learning frameworks, but this is obviously not an attempt to replicate pytorch. Classes are used to demonstrate separate concepts involved in creating a model, rather than for development and maintenance of the framework.

## Model

A model is the primary goal a deep learning. The model is able to learn from data in order to make predictions. A model is a collection of 'layers', each of which represents a set of linear and non-linear transformations. In a way, the model is the set of parameters that define the layer, but it is also the data used to train it and the optimization strategy used to arrive at those parameters through the data.

## Dataset

An abstraction to contain the loading, splitting, and preprocessing of data.

## Layer

The basic unit of a neural net, each layer is a set of weights, biases, and activation functions that transform a set of inputs into outputs.

## DifferentialbeFunction

Differentialbe functions are used in activations and losses. Metrics are distinguished by the fact that they do not need to be differentiable.

## Optimizer

The optimization strategy used to update the parameters of each layer.

## Training

A training loop involves the 'forward pass' that results in a prediction given the models current parameters. Then the loss is calculated to quantify how far from the correct output the predictions are. Next a 'backward' pass is used to calculate the derivatives with respect to the parameters that make up the layer. The optimization strategy updates the parameters in such a way that the next prediction will hopefully be closer to the correct values. So each loop has
- Forward pass
- Calculate loss
- Backward pass
- Optimization