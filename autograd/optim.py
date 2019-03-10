"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.nodes


####################
#### OPTIMIZERS ####
####################


class Adam(autograd.nodes.Optimizer):

    def __init__(self, alpha=0.0001, beta_one=0.9, beta_two=0.999, epsilon=1e-8):
        """Creates an ADAM optimizer."""
        super(Adam, self).__init__("adam")
        self.t = 0
        self.alpha = alpha
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def forward(self, variable):
        """Computes the result of this operation."""
        if self.m is None:
            self.m = np.zeros(variable.shape)
        if self.v is None:
            self.v = np.zeros(variable.shape)
        return variable.data

    def backward(self, gradient, variable):
        """Computes the gradient with respect to *args."""
        self.t += 1
        self.m = self.beta_one * self.m + (1 - self.beta_one) * gradient
        self.v = self.beta_two * self.v + (1 - self.beta_two) * gradient**2
        m_hat = self.m / (1 + self.beta_one**self.t)
        v_hat = self.v / (1 + self.beta_two**self.t)
        return [self.alpha * m_hat / np.sqrt(v_hat + self.epsilon)]
