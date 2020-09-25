# import standard modules
from copy import deepcopy
import random

# import third party modules
import numpy as np

# import project related modules
from Network.activation import Softmax


class Dense(object):

    def __init__(self, neurons: int, size: int, learning_rate=0.01, input_layer=False, activation=None):
        self.neurons = neurons

        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self._initialize_weights(size)
        self.cache = None
        self.activation = {"softmax": Softmax(), None: None}[activation]
        self.input_layer = input_layer

    def __str__(self):
        return f"Dense: {self.neurons}"

    def _initialize_weights(self, size):
        self.weights = np.random.randn(size, self.neurons)  # * np.sqrt((2/size)) * 0.1
        self.bias = np.zeros((1, self.neurons))

    def forward(self, x, skip_bias=True):

        if self.input_layer:
            x = x.T
        self.cache = deepcopy(x)
        if self.weights is None:
            self._initialize_weights(x.shape[1])

        self.bias = 0 if skip_bias else self.bias
        result = np.dot(x, self.weights) + self.bias

        if self.activation is None:
            return result
        else:
            return self.activation.forward(result)

    def backward(self, z):

        # print(f"in: {self.cache.shape}, w: {self.weights.shape}, z: {z.shape}")
        a_delta = np.dot(z, self.weights.T)
        w_delta = np.dot(self.cache.T, z)
        b_delta = np.sum(z, axis=1, keepdims=True)

        self.weights -= (self.learning_rate * w_delta)
        self.bias -= (self.learning_rate * b_delta)

        return a_delta