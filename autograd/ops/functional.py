"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.nodes


####################
#### OPERATIONS ####
####################


def sigmoid_backend(x):
    """Applies the sigmoid function elementwise."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_gradient_backend(x):
    """Applies the sigmoid gradient function elementwise."""
    return sigmoid_backend(x) * (1 - sigmoid_backend(x))


class Sigmoid(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Sigmoid, self).__init__("sigmoid", sigmoid_backend, 
            lambda g, x: g * sigmoid_gradient_backend(x))


def sigmoid(x):
    """Applies the Sigmoid function elementwise."""
    operation = Sigmoid()
    operation.resolve(x)
    return operation


def tanh_backend(x):
    """Applies the tanh function elementwise."""
    return np.tanh(x)


def tanh_gradient_backend(x):
    """Applies the tanh gradient function elementwise."""
    return (1.0 - tanh_backend(x)**2)


class Tanh(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Tanh, self).__init__("tanh", tanh_backend, 
            lambda g, x: [g * tanh_gradient_backend(x)])


def tanh(x):
    """Applies the Tanh function elementwise."""
    operation = Tanh()
    operation.resolve(x)
    return operation


def relu_backend(x):
    """Applies the relu function elementwise."""
    return np.where(x >= 0, x, 0.0)


def relu_gradient_backend(x):
    """Applies the relu gradient function elementwise."""
    return np.where(x >= 0, 1.0, 0.0)


class Relu(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(Relu, self).__init__("relu", relu_backend, 
            lambda g, x: [g * relu_gradient_backend(x)])


def relu(x):
    """Applies the Relu function elementwise."""
    operation = Relu()
    operation.resolve(x)
    return operation


def log_sigmoid_backend(x):
    """Applies the log sigmoid function elementwise."""
    return np.where(x >= 0, -np.log(1 + np.exp(-x)), x - np.log(1 + np.exp(x)))
    

def log_sigmoid_gradient_backend(x):
    """Applies the log sigmoid gradient function elementwise."""
    return np.where(x >= 0, np.exp(-x) / (1 + np.exp(-x)), 1 / (1 + np.exp(x)))


def sigmoid_cross_entropy_backend(a, b):
    """Computes a sparse cross entropy loss function."""
    return -np.mean(a * log_sigmoid_backend(b) + 
        np.subtract(1.0, a) * log_sigmoid_backend(np.subtract(1.0, b)))


def sigmoid_cross_entropy_gradient_backend(g, a, b):
    """Computes a sparse cross entropy gradient."""
    return [
        (-g * (log_sigmoid_backend(b) - log_sigmoid_backend(np.subtract(1.0, b))) / np.size(a)), 
        (-g * (a * log_sigmoid_gradient_backend(b) -
            np.subtract(1.0, a) * log_sigmoid_gradient_backend(np.subtract(1.0, b))) / np.size(a))]


class SigmoidCrossEntropy(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        super(SigmoidCrossEntropy, self).__init__("sigmoid_cross_entropy", 
            sigmoid_cross_entropy_backend, sigmoid_cross_entropy_gradient_backend)


def sigmoid_cross_entropy(a, b):
    """Computes a sigmoid cross entropy loss function."""
    operation = SigmoidCrossEntropy()
    operation.resolve(a, b)
    return operation


class SigmoidClassifier(autograd.nodes.Operation):

    def __init__(self, threshold):
        """Creates a graph node in a dynamic comptational graph."""
        super(SigmoidClassifier, self).__init__("sigmoid_classifier", 
            lambda x: np.where(threshold > sigmoid_backend(x), 0, 1), 
            lambda g, x: np.zeros(x.shape))


def sigmoid_classifier(x, threshold):
    """Computes a classifier using the sigmoid function."""
    operation = SigmoidClassifier(threshold)
    operation.resolve(x)
    return operation


