"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import scipy.io as io
import numpy as np
import autograd.utils
import autograd.layers


#####################
#### ENTRY POINT ####
#####################


if __name__ == "__main__":

    dataset = io.loadmat("data.mat")

    training_data = dataset["X"]
    training_labels = dataset["y"]

    VAL_SEED = 189

    np.random.seed(VAL_SEED)
    np.random.shuffle(training_data)
    np.random.seed(VAL_SEED)
    np.random.shuffle(training_labels)

    VAL_SIZE = 500

    validation_data = autograd.utils.constant(training_data[VAL_SIZE:, :])
    validation_labels = autograd.utils.constant(training_labels[VAL_SIZE:, :])
    training_data = autograd.utils.constant(training_data[:VAL_SIZE, :])
    training_labels = autograd.utils.constant(training_labels[:VAL_SIZE, :])

    # Create the neural network
    layer1 = autograd.layers.Dense(12, 4, activation=autograd.utils.tanh)
    layer2 = autograd.layers.Dense(4, 1)

    def get_val_accuracy(threshold):
        """Get the accuracy of the curent model."""
        return np.mean(np.where(
            np.equal(
                autograd.utils.sigmoid_classifier(layer2(layer1(validation_data)), threshold).data, 
                validation_labels.data), 
            1.0, 0.0))


    # Loop through the dataset
    for i in range(100000):

        # Compute a forward pass
        x = layer1(training_data)
        x = layer2(x)

        # Compute the loss and backpropagate
        loss = autograd.utils.sigmoid_cross_entropy(training_labels, x)
        penalty = autograd.utils.weight_decay(loss, 0.001)
        loss = autograd.utils.add(loss, penalty)
        autograd.utils.backpropagate(loss)

        max_threshold = 0.5
        max_accuracy = get_val_accuracy(max_threshold)

        if loss.data < 0.05:

            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                accuracy = get_val_accuracy(threshold)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_threshold = threshold

        print("Loss was {0:.5f} Accuracy was {1:.5f} with threshold {2:.2f}.".format(
            loss.data, max_accuracy, max_threshold))
