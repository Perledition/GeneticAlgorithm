# import standard modules

# import third party modules
import numpy as np

# import project related modules


class CostFunction(object):

    def __init__(self):
        self.loss = list()
        self.error = None
        self.cache = None
        self.target = None


class CrossEntropy(CostFunction):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        x: prediction with applied softmax
        y: target vector
        In order to calculate the cross entropy the H(p, q) formular is applied -SUM(p(x)log q(x))
        with p equals target class and q equals the predicted class
        """

        self.cache = x.copy()
        self.target = y.copy()
        epsilon = 1e-12

        predictions = np.clip(self.cache, epsilon, 1. - epsilon)
        N = self.cache.shape[0]
        cost = -np.sum(self.target*np.log(self.cache+1e-9))/N

        assert(isinstance(cost, float))

        return cost

    def backward(self):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector.
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = self.target.shape[0]

        index = np.argmax(self.target)

        self.cache[:m, np.argmax(self.target)] -= 1
        self.cache = self.cache/m
        return self.cache


class Metrics:

    def __init__(self):
        pass


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