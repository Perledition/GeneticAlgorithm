import numpy as np
from collections import Counter


class ClassificationMatrix:

    def __init__(self):
        self.matrix = None

    def print_matrix(self):
        pass

    def _calculate_matrix(self, y_hat, y):
        result = dict()
        for i in range(0, 4):
            y_indexes = list(np.where(y == i)[0])

            collection = list(zip(list(y[y_indexes]), list(y_hat[y_indexes])))
            c = Counter(elem for elem in collection)
            result[i] = dict(zip([x[1] for x in list(c.keys())], [c[ele] for ele in list(c.keys())]))

        return np.array([
            [self.value_pick(values, cl) for classes, values in result.items()]
            for cl in range(0, 4)]
        )

    @staticmethod
    def value_pick(dict_map, key):
        try:
            return dict_map[key]
        except KeyError:
            return 0

    @staticmethod
    def class_size(y):

        if y.shape[1] == 1:
            classes = [0, 1]
        else:
            classes = [c for c in range(y.shape)]
        return classes

    def fit(self, y_hat, y):

        # ensure equal shape of target variables and predicted values
        assert y_hat.shape == y.shape, f"Shapes of predictions and targets do not fit {y_hat.shape} != {y.shape}"

        self.matrix = self._calculate_matrix(y_hat, y)
        return self.matrix