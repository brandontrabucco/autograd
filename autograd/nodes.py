"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.core


###############
#### NODES ####
###############


class Constant(autograd.core.GraphNode):

    def __init__(self, value):
        """Creates a graph node in a dynamic comptational graph."""
        super(Constant, self).__init__("constant")
        self.data = value
        self.inputs = []

    def backward(self, gradient):
        """Computes the gradient with respect to *args."""
        # Nothing to see here
        return []


class Variable(autograd.core.GraphNode):

    def __init__(self, value):
        """Creates a graph node in a dynamic computational graph."""
        super(Variable, self).__init__("variable")
        self.data = value
        self.inputs = []

    def backward(self, gradient):
        """Computes the gradient with respect to *args."""
        self.data -= gradient
        return []


class Operation(autograd.core.GraphNode):

    def __init__(self, name, forward_function, backward_function):
        """Creates a graph node in a dynamic comptational graph."""
        super(Operation, self).__init__(name + "_operation")
        self.forward_function = forward_function
        self.backward_function = backward_function

    def forward(self, *inputs):
        """Computes the result of this operation."""
        return self.forward_function(*inputs)

    def backward(self, gradient, *inputs):
        """Computes the gradient with respect to *inputs."""
        return self.backward_function(gradient, *inputs)


class Optimizer(autograd.core.GraphNode):

    def __init__(self, name):
        """Creates a graph node in a dynamic comptational graph."""
        super(Optimizer, self).__init__(name + "_optimizer")

    def forward(self, variable):
        """Computes the result of this operation."""
        return variable.data

    def backward(self, gradient, variable):
        """Computes the gradient with respect to *args."""
        return [gradient]
