# import standard modules

# import third party modules
import numpy as np

# import project related modules


class CostFunction(object):
    """
    Parent class for all cost Functions within NumpyNet. It is used to reduce the amount of code for initialization
    since all cost functions share the same attributes.
    """

    def __init__(self):
        self.loss = list()   # list of losses produced
        self.error = None    # error between x and y values
        self.cache = None    # copy of input data x
        self.target = None   # copy of input data y


class CrossEntropy(CostFunction):
    """
    cross entropy used as loss function for a classification problem with n possible classes with a one hot encoded
    vector for y and predicted values. CrossEntropy inherits from CostFunction class.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: np.array, y: np.array):
        """
        to calculate the cross entropy the H(p, q) formula is applied. with p being y and q being x.
        and x is a result coming from an applied Softmax(x) function.

        :param: x: numpy.array: predicted data x with applied Softmax function and in shape of a one hot encoded vector
        :param: y: numpy.array: target variables (label) in a shape of a one hot encoded vector

        :return: float: absolute value of cost
        """

        # create a copy of x and y inputs in order to make them available for the error definition step
        self.cache = x.copy()
        self.target = y.copy()

        # get the amount of samples to calculate
        # calculate the cost the with the given H(p, q formula)
        N = self.cache.shape[0]
        cost = -np.sum(self.target*np.log(self.cache+1e-9))/N
        return cost

    def backward(self):
        """
        calculates the error between predicted values and target values. It will basically return all values
        as they are but subtracts 1 from the index which should be the target class.
        """

        # get amount of samples
        m = self.target.shape[0]

        # subtract one from the index where the highest probability should be divide the error values by the amount
        # of samples and return the error array
        self.cache[:m, np.argmax(self.target)] -= 1
        self.cache = self.cache/m
        return self.cache


class Accuracy(Metrics):

    def __init__(self):
        super().__init__()
        self.values = list()

    def add(self, x, y):
        pred = np.argmax(x)
        target = np.argmax(y)

        if int(pred)-int(target) == 0:
            self.values.append(1)
        else:
            self.values.append(0)

    def clean(self):
        self.values = list()

    def get(self):
        return f"accuracy: {sum(self.values)/len(self.values) * 100}%"