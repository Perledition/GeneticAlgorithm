# import standard modules
from collections import Counter

# import third party modules
import numpy as np

# import project related modules


class NaiveClassificationMatrix:
    """
    A simplified version of NumpyNets ClassificationMatrix class. The main difference is the handling of dimensions
    is order to reduce the work of generalization for the Genetic use case.

    Creates a classification matrix as follows:
                        predicted class 1 | predicted class 2 | predicted n
        label class 1:          12        |        0          |      ...
        label class 1:          0         |        32          |      ...

    Therefore the label is produced as rows while the predicted output is on the column space.
    However for use outside the use case of GeneticAlgorithms it's strongly suggested to go with the
    ClassificationMatrix from NumpyNet.

    """
    def __init__(self):
        self.matrix = None  # placeholder fot the classification matrix as numpy array

    def _calculate_matrix(self, y_hat: np.array, y: np.array):
        """
        class internal function to calculate a simple classification matrix in order over a given prediction array
        and it's true label array.

        :param y_hat: numpy.array: array of predictions
        :param y: numpy.array: array of labels

        :return: numpy.array: classification matrix
        """

        # initialize an empty array in order to collect the results for each combination
        result = dict()

        # count the amount of unique classes
        unique_classes = len(list(set(y)))

        # run the loop for each unique class from the label array
        for i in range(0, unique_classes):

            # find all indexes in the label array which fit the current class
            y_indexes = list(np.where(y == i)[0])

            # get these indexes from the label array and from the prediction array and create a
            # tuple (label value at index, prediction value at index) for each index.
            collection = list(zip(list(y[y_indexes]), list(y_hat[y_indexes])))

            # count how often each combination appears
            c = Counter(elem for elem in collection)

            # create a dict entry with the class value as key and another dict of predicted classes and counts
            # e.g. class 0: {class 0: 12, class 1: 3, .... class n: n}
            result[i] = dict(zip([x[1] for x in list(c.keys())], [c[ele] for ele in list(c.keys())]))

        # decode the dict in order to get the classification matrix
        # again. the Classification Matrix from NumpyNet is in a better shape and also more effective
        return np.array([
            [self.value_pick(values, cl) for classes, values in result.items()]
            for cl in range(0, unique_classes)]
        )

    @staticmethod
    def value_pick(dict_map: dict, key: int):
        """
        static function which tries to get a key from a given dict and return 0 if key error appears.

        :param dict_map: dict: dictionary to pick the key from
        :param key: int: key to access

        :return: value at dict_map[key] or 0 if KeyError appears
        """

        try:
            return dict_map[key]
        except KeyError:
            return 0

    def fit(self, y_hat: np.array, y: np.array):
        """
        fit function produces the classification matrix for a given array of predictions and a given array of labels.
        NaiveClassificationMatrix does only handle 1d arrays

        :param y_hat: numpy.array: array of predictions
        :param y: numpy.array: array of labels

        :return: numpy.array: classification matrix
        """
        # ensure equal shape of target variables and predicted values
        assert y_hat.shape == y.shape, f"Shapes of predictions and targets do not fit {y_hat.shape} != {y.shape}"

        # create classification matrix and return it
        self.matrix = self._calculate_matrix(y_hat, y)
        return self.matrix
