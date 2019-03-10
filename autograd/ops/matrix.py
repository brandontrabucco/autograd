"""Author: Brandon Trabucco, Copyright 2019
Implements dynamic computational graphs with an interface like pytorch.
Also uses the ADAM optimizer."""


import numpy as np
import autograd.nodes


####################
#### OPERATIONS ####
####################


def transpose_backend(x):
    """Transpose a stack of matrices over the last dimension."""
    return np.transpose(x, list(range(len(x.shape) - 2)) + [-1, -2])


class Matmul(autograd.nodes.Operation):

    def __init__(self):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda a, b:  np.matmul(a, b)
        backward_function = lambda g, a, b: [np.matmul(g, transpose_backend(b)), 
            np.matmul(transpose_backend(a), g)]
        super(Matmul, self).__init__("matmul", forward_function, backward_function)


def matmul(a, b):
    """Matrix multiplies two stacked matrices."""
    operation = Matmul()
    operation.resolve(a, b)
    return operation


class Broadcast(autograd.nodes.Operation):

    def __init__(self, axis):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda x: x
        backward_function = lambda g, x: [np.sum(g, axis=axis, keepdims=True)]
        super(Broadcast, self).__init__("broadcast", forward_function, backward_function)


def broadcast(x, axis):
    """Allows for a tensor to broadcast along an axis."""
    operation = Broadcast(axis)
    operation.resolve(x)
    return operation


class SquaredTwoNorm(autograd.nodes.Operation):

    def __init__(self, weight):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda x: weight * np.sum(np.square(x))
        backward_function = lambda g, x: [2.0 * weight * g * x]
        super(SquaredTwoNorm, self).__init__("squared_two_norm", forward_function, backward_function)


def squared_two_norm(x, weight):
    """Computes the two norm of a tensor."""
    operation = SquaredTwoNorm(weight)
    operation.resolve(x)
    return operation


def bytes_offset_backend(x):
    """Get offset of array data from base data in bytes."""
    return (0 if x.base is None else 
        x.__array_interface__['data'][0] - x.base.__array_interface__['data'][0])


def calc_size_backend(size, kernel_size, padding, stride):
    """Calculate the size of one image dimension."""
    return int(np.ceil((size - kernel_size + padding + 1) / stride))


def sliding_window2d_backend(x, kernel_size, padding, stride):
    """Converts a tensor to sliding windows."""
    padded_x = np.pad(x, [[0, 0], 
        [int(np.floor(padding[0] / 2)), int(np.ceil(padding[0] / 2))], 
        [int(np.floor(padding[1] / 2)), int(np.ceil(padding[1] / 2))], 
        [0, 0]], mode='constant', constant_values=(0.0,))
    return np.ndarray([x.shape[0], 
        calc_size_backend(x.shape[1], kernel_size[0], padding[0], stride[0]), 
        calc_size_backend(x.shape[2], kernel_size[1], padding[1], stride[1]), 
        kernel_size[0], kernel_size[1], x.shape[3]],
        dtype=padded_x.dtype, buffer=padded_x.data, offset=bytes_offset_backend(padded_x),
        strides=[x.strides[0], stride[0] * x.strides[1], stride[1] * x.strides[2], 
            x.strides[1], x.strides[2], x.strides[3]])


def conv2d_backend(x, filters, padding, stride):
    """Performs a discrete 2d convolution of x with filters."""
    x = sliding_window2d_backend(x, filters.shape[:2], padding, stride)
    return np.matmul(
        x.reshape([x.shape[0] * x.shape[1] * x.shape[2], x.shape[3] * x.shape[4] * x.shape[5]]), 
        np.reshape(filters, [filters.shape[0] * filters.shape[1] * filters.shape[2], filters.shape[3]])
    ).reshape([x.shape[0], x.shape[1], x.shape[2], filters.shape[3]])


def conv2d_grad_backend(x, dy, padding, stride):
    """Performs the gradient of a discrete 2d convolution of x with w."""
    dilated_dy = np.zeros([dy.shape[0], dy.shape[1], stride[0], dy.shape[2], stride[1], dy.shape[3]])
    dilated_dy[:, :, 0, :, 0, :] = dy
    dy = dilated_dy.reshape([dy.shape[0], dy.shape[1] * stride[0], dy.shape[2] * stride[1], dy.shape[3]])
    return np.transpose(conv2d_backend(np.transpose(x, [3, 1, 2, 0]), 
        np.transpose(dy, [1, 2, 0, 3]), padding, [1, 1]), [1, 2, 0, 3])


def conv2d_transpose_backend(x, filters, padding, stride):
    """Performs a discrete transpose 2d convolution of x with filters."""
    dilated_x = np.zeros([x.shape[0], x.shape[1], stride[0], x.shape[2], stride[1], x.shape[3]])
    dilated_x[:, :, 0, :, 0, :] = x
    x = dilated_x.reshape([x.shape[0], x.shape[1] * stride[0], x.shape[2] * stride[1], x.shape[3]])
    return conv2d_backend(x, np.transpose(filters, [0, 1, 3, 2]), padding, [1, 1])


class Conv2d(autograd.nodes.Operation):

    def __init__(self, padding, stride):
        """Creates a graph node in a dynamic comptational graph."""
        forward_function = lambda x, w: conv2d_backend(x, w, padding, stride)
        backward_function = lambda g, x, w: [conv2d_transpose_backend(g, w, padding, stride), 
            conv2d_grad_backend(x, g, padding, stride)]
        super(Conv2d, self).__init__("conv2d", forward_function, backward_function)


def conv2d(x, w, padding, stride):
    """Computes a differentiable convolution operation."""
    operation = Conv2d(padding, stride)
    operation.resolve(x, w)
    return operation
