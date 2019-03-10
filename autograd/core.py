"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np


##############
#### CORE ####
##############


class GraphNode(object):

    def __init__(self, name):
        """Creates a graph node in a dynamic comptational graph."""
        self.name = name
        self.inputs = None
        self.data = None
        self.gradient = None

    def resolve(self, *inputs):
        """Computes the result of this operation."""
        self.inputs = inputs
        self.data = self.forward(*[x.data for x in self.inputs])
        return self.data

    def update(self, gradient):
        """Computes the gradient with respect to inputs."""
        self.gradient = gradient
        for x, grad_x in zip(self.inputs, self.backward(gradient, *[x.data for x in self.inputs])):
            x.update(grad_x)

    def forward(self, *inputs):
        """Computes the result of this operation."""
        raise NotImplemented

    def backward(self, gradient, *inputs):
        """Computes the gradient with respect to inputs."""
        raise NotImplemented
