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
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, input_size, output_size, activation=lambda x: x):
        """Creates a dense neural network layer."""
        super(Dense, self).__init__("dense")
        self.weights = autograd.utils.variable([input_size, output_size])
        self.bias = autograd.utils.variable([1, output_size])
        self.activation = activation

    def __call__(self, x):
        """Performs a forward pass for a neural net layer."""
        return self.activation(autograd.utils.add(
            autograd.utils.matmul(x, self.weights), 
            autograd.utils.broadcast(self.bias, 0)))


class Conv2d(Layer):

    def __init__(self, kernel_size, input_size, output_size, padding, stride, activation=lambda x: x):
        """Creates a convolutional neural network layer."""
        super(Conv2d, self).__init__("conv2d")
        self.weights = autograd.utils.variable(kernel_size + [input_size, output_size])
        self.bias = autograd.utils.variable([1, 1, 1, output_size])
        self.padding = padding
        self.stride = stride
        self.activation = activation

    def __call__(self, x):
        """Performs a forward pass for a neural net layer."""
        return self.activation(autograd.utils.add(
            autograd.utils.conv2d(x, self.weights, self.padding, self.stride), 
            autograd.utils.broadcast(self.bias, [0, 1, 2])))
