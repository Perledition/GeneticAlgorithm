# import standard modules
from copy import deepcopy

# import third party modules
import numpy as np

# import project related modules
from simple_numpy_net.general import Layer
from simple_numpy_net.activation import Softmax, Sigmoid


class Dense(Layer):
    """
    A regular densely-connected layer. It can be used as input layer, output layer or hidden layer by setting the
    parameters accordingly. Dense inherits form Layer

    :param: neurons: int: Positive integer, dimensionality of the output space.
    :param: learning_rate: float: defines the learning rate for the layers back propagation step (default 0.01)
    :param: input_layer: boolean: True if it's the first densely connected layer in the network receiving raw input
    :param: activation: str: name of activation function from output layer. only "softmax" and "sigmoid" supported,
                             default None which means no probability distribution over output is generated.

    """

    def __init__(self, neurons: int, learning_rate: float = 0.01,
                 input_layer: bool = False, activation: str = None):
        super().__init__()                           # call to parent class
        self.neurons = neurons                       # neurons dimension of the layer
        self.learning_rate = learning_rate           # defines learning rate of the layer
        self.weights = None                          # placeholder for weight matrix
        self.bias = None                             # placeholder for bias matrix
        self.input_layer = input_layer               # bool definition whether layer is input layer

        # self activation defines how and if a probability distribution is generated from the predictions
        self.activation = {"softmax": Softmax(), "sigmoid": Sigmoid(), None: None}[activation]

    def __str__(self):
        """
        string representation of the layer which displays the most important information of the layer
        :return: str: string with layer information
        """

        # create string value of bool value
        input_l = "True" if self.input_layer else "False"

        # create string tag for Sigmoid, Softmax or NoneType class
        if isinstance(self.activation, Sigmoid):
            dist = "sigmoid"
        elif isinstance(self.activation, Softmax):
            dist = "softmax"
        else:
            dist = "None"

        # bring together information in one string and return the string value
        return f"Dense: neurons: {self.neurons}\n learning rate: {self.learning_rate}, input layer: {input_l}, " \
               f"probability distribution: {dist}"

    def _initialize_weights(self, size: int):
        """
        class internal function to initialize weights and bias. The shape is determined by the amount of neurons
        and the incoming shape of data from the previous layer or raw data. Weights get initialized randomly. However,
        a smoothing technique is used as well as a multiplication with 0.1 since we want the starting weights to be
        quite small.

        :param size: int: amount of incoming values from the previous layer or raw data
        :return: None
        """

        # uses He Normal or He-at-el initialization become famous by a paper 2015 by He-et-al
        self.weights = np.random.randn(size, self.neurons) * np.sqrt((2/size)) * 0.1
        self.bias = np.zeros((1, self.neurons))

    def forward(self, x: np.array):
        """
        forward process of the layer. Takes values x and performs a linear function operation on the data.
        operation: x * weights + bias. If layer has an activation function the results will be smoothed over
        a probability distribution, otherwise the the results will be returned as they come from the operation.

        :param x: numpy.array: array of x which should taken into account for the forward operation
        :return: numpy.array: operation results
        """

        # checks if input layer. If this is the case transpose the data in order to ensure the horizontal orientation
        # of the incoming data
        if self.input_layer:
            x = x.T

        # copy incoming data and keep it as global attribute in order to perform a back propagation later on
        self.cache = deepcopy(x)

        # in case layer weights are not initialized yet create them and the bias before executing the forward pass
        if self.weights is None:
            self._initialize_weights(x.shape[1])

        # run the matrix multiplication and add the bias
        result = np.dot(x, self.weights) + self.bias

        # apply the function to create a probability distribution in case activation function is not None
        # otherwise return the raw results
        if self.activation is None:
            return result
        else:
            return self.activation.forward(result)

    def backward(self, z: np.array):
        """
        function to perform the back propagation step for this layer. It takes an error z and returns a new error array,
        after changing the attributes for this layer accordingly.

        :param z: numpy.array: array of error values for the back propagation
        :return: numpy.array: returns modified error values with respect to the weights of this layer
        """

        # create new error values which will be returned
        a_delta = np.dot(z, self.weights.T)

        # create the delta for weights and bias
        # while weights delta is a matrix multiplication of the original input data and the error term
        # the bias delta is the sum of each row
        w_delta = np.dot(self.cache.T, z)
        b_delta = np.sum(z, axis=1, keepdims=True)

        # apply the delta to a degree of the learning rate to the weights and bias
        # syntax: -= -> e.g. weights = weights - (alpha * delta)
        self.weights -= (self.learning_rate * w_delta)
        self.bias -= (self.learning_rate * b_delta)

        # return delta
        return a_delta