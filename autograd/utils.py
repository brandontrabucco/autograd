"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.nodes
import autograd.optim
from autograd.ops.core import mean, add, subtract, multiply, divide
from autograd.ops.matrix import matmul, broadcast, squared_two_norm, conv2d
from autograd.ops.functional import sigmoid, tanh, relu, sigmoid_cross_entropy, sigmoid_classifier


###############
#### UTILS ####
###############


def backpropagate(x):
    """Push gradients back through the graph."""
    x.update(np.ones(x.data.shape))


def variable(shape, optimizer=autograd.optim.Adam, **kwargs):
    """Create a variable with the specified shape."""
    var = autograd.nodes.Variable(np.random.normal(0, 0.01, shape))
    optimized_var = optimizer(**kwargs)
    optimized_var.resolve(var)
    return optimized_var


def constant(value):
    """Create a constant with the specified value."""
    return autograd.nodes.Constant(value)


def weight_decay(x, weight, penalty_function=squared_two_norm):
    """Recurse from x and decay all weights."""
    if isinstance(x, autograd.nodes.Optimizer):
        return penalty_function(x, weight)
    elif len(x.inputs) == 0:
        return constant(np.array(0.0))
    value = weight_decay(x.inputs[0], weight)
    for y in x.inputs[1:]:
        value = add(value, weight_decay(y, weight))
    return value
    