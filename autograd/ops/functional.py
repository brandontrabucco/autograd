"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.nodes


####################
#### OPERATIONS ####
####################


class Sigmoid(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda x: np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        backward_function = lambda g, x: [g * forward_function(x) * (1 - forward_function(x))]
        super(Sigmoid, self).__init__("sigmoid", forward_function, backward_function)


def sigmoid(x):
    """Applies the Sigmoid function elementwise."""
    operation = Sigmoid()
    operation.resolve(x)
    return operation


class Tanh(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda x: np.tanh(x)
        backward_function = lambda g, x: [g * (1.0 - forward_function(x))**2]
        super(Tanh, self).__init__("tanh", forward_function, backward_function)


def tanh(x):
    """Applies the Tanh function elementwise."""
    operation = Tanh()
    operation.resolve(x)
    return operation


class Relu(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda x: np.where(x >= 0, x, 0.0)
        backward_function = lambda g, x: [np.where(x >= 0, 1.0, 0.0)]
        super(Relu, self).__init__("relu", forward_function, backward_function)


def relu(x):
    """Applies the Relu function elementwise."""
    operation = Relu()
    operation.resolve(x)
    return operation
