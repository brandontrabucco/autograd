"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.utils
import autograd.layers


#####################
#### ENTRY POINT ####
#####################


if __name__ == "__main__":

    # Create the neural network
    layer1 = autograd.layers.Dense(1024, 1024)
    layer2 = autograd.layers.Dense(1024, 1024)
    layer3 = autograd.layers.Dense(1024, 1024)

    # Loop through the dataset
    for i in range(10):

        # Load a batch of training examples
        x = autograd.utils.constant(np.ones([256, 1024]))

        # Compute a forward pass
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)

        # Compute the loss and backpropagate
        loss = autograd.utils.mean(x)
        autograd.utils.backpropagate(loss)
        print(loss.data)
