"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import autograd.utils


################
#### LAYERS ####
################


class Layer(object):

    def __init__(self, name):
        """Creates a neural network layer."""
        self.name = name + "_layer"

    def __call__(self, x):
        """Performs a forward pass for a neural net layer."""
        raise NotImplemented


class Dense(Layer):

    def __init__(self, input_size, output_size):
        """Creates a dense neural network layer."""
        super(Dense, self).__init__("dense")
        self.weights = autograd.utils.variable([input_size, output_size])
        self.bias = autograd.utils.variable([1, output_size])

    def __call__(self, x):
        """Performs a forward pass for a neural net layer."""
        return autograd.utils.tanh(autograd.utils.add(
            autograd.utils.matmul(x, self.weights), 
            autograd.utils.broadcast(self.bias, 0)))
