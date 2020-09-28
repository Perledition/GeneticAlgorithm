# import standard modules

# import third party modules
import numpy as np

# import project related modules
from simple_numpy_net.general import Layer


class Sigmoid(Layer):
    """
    create a sigmoid function which can be used within a neuronal network due to the fact that also the derivative
    is available. Sigmoid inherits form Layer
    """

    def __init__(self):
        super().__init__()  # call to parent class

    def __str__(self):
        """
        string representation of the layer.

        :return: str: returns the name of the activation function and it's function
        """
        return f"Activation: Sigmoid"

    def forward(self, x: np.array):
        """
        performs a sigmoid(x) operation of parameter x.

        :param x: numpy.array: data which does into the sigmoid function
        :return: numpy.array of sigmoid(x)
        """

        # line of code to prevent overflow in numpy since this is a common issue
        signal = np.clip(x, -150000, 150000)

        # perform the sigmoid operation with the overflow signal and data x
        # safe the result in global attribute cache, since the result will be needed for back propagation as well
        self.cache = 1.0 / (1.0 + np.exp(-1.0 * signal))
        return self.cache

    def backward(self, z: np.array):
        """
        takes an array with error values and performs a back propagation step for the Sigmoid activation function.
        will will return a new error array with respect to the Sigmoid function.

        :param z: np.array: array of error values
        :return: returns the derivative of sigmoid with respect to it's previous calculated output and error z
        """

        return (self.cache * (1.0 - self.cache)) * z


class RelU(Layer):
    """"
    RelU or Rectified Linear Unit function as activation function which will output the input directly if it's
    positive or put it to zero if not. RelU inherits form Layer
    """

    def __init__(self):
        super().__init__()  # call to parent class

    def __str__(self):
        """
        string representation of the activation layer and it's function
        :return: str: Type and Function
        """
        return "Activation: RelU"

    def forward(self, x: np.array):
        """
        RelU(x) operation - will out put the same array but with values < 0 as 0
        :param x: numpy.array: data to which goes into the RelU function

        :return: numpy.array: returns the numpy array with modified values < 0
        """

        # make a copy of x for the back propagation process
        self.cache = x.copy()

        # apply function and return new array
        return np.maximum(0, x)

    def backward(self, z: np.array):
        """
        creates a derivative of the RelU with respect to it's input and multiplies it with the incoming error array.

        :param z: numpy.array: array of error values to be back propagated
        :return: numpy.array: returns an array of new errors with the RelU layer taken into account
        """

        # as usual values smaller 0 will be set to zero. But since it's the derivative values greater 0 will be set to
        # 1 and the error is multiplied in order to propagate the error back.
        return np.where(self.cache < 0, 0, 1) * z


class Softmax:
    """
    provides only one function with is forward since backward is not needed. Produces a probability distribution over
    k input values.
    """

    def forward(self, x: np.array):
        """
        performs a Softmax(x) operation.
        :param x: numpy.array: data which will be feed into the Softmax function
        :return: numpy.array for a smooth probability distribution over input x
        """

        # operation to avoid overflow in numpy
        x = np.clip(x, -150000, 150000)

        # create an exponential of x and subtract the max value of x in order to make the function more stable
        exps = np.exp(x - np.max(x))

        # return results of softmax function
        return exps / np.sum(exps)