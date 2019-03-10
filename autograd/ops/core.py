"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.nodes


####################
#### OPERATIONS ####
####################


class Mean(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Mean, self).__init__("mean", lambda x: np.mean(x), lambda g, x: [g * np.ones(x.shape) / x.size])


def mean(x):
    """Reduce a tensor to a scalar using the mean."""
    operation = Mean()
    operation.resolve(x)
    return operation


class Add(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Add, self).__init__("add", lambda a, b: a + b, lambda g, a, b: [g, g])


def add(a, b):
    """Wrapper for the Add operation."""
    operation = Add()
    operation.resolve(a, b)
    return operation


class Subtract(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Subtract, self).__init__("subtract", lambda a, b: a - b, lambda g, a, b: [g, -g])


def subtract(a, b):
    """Wrapper for the Subtract operation."""
    operation = Subtract()
    operation.resolve(a, b)
    return operation


class Multiply(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Multiply, self).__init__("multiply", lambda a, b: a * b, lambda g, a, b: [g * b, g * a])


def multiply(a, b):
    """Wrapper for the Multiply operation."""
    operation = Multiply()
    operation.resolve(a, b)
    return operation


class Divide(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Divide, self).__init__("divide", lambda a, b: a / b, lambda g, a, b: [g / b, -g * a / b**2])


def divide(a, b):
    """Wrapper for the Divide operation."""
    operation = Divide()
    operation.resolve(a, b)
    return operation
