# import standard modules
import inspect
from collections import Counter

# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# import project related modules
from Network.dense import Dense


class ConfusionMatrix:

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


# TODO: USE AUC as Fitness function since it provides the max character and allows easy calculation
class GeneticSequence(object):

    def __init__(self, layer: list, model_type="classification"):
        self.layer = layer
        self.fitness_func = {"classification": ConfusionMatrix, "regression": 1}[model_type]

    def get_all_weights(self):
        return [l.weights for l in self.layer if isinstance(l, Dense)]

    def update_weights(self, weights):
        pass

    def calculate_fitness(self, y_hat, y):

        matrix = ConfusionMatrix().fit(y_hat, y)
        return matrix

    def predict(self, x):

        results = list()
        for ix in range(0, x.shape[0]):

            # forward feed
            z = x[ix].reshape(-1, 1)
            for i, layer in enumerate(self.layer):
                z = self.layer[i].forward(z)

            # append results to list of results
            results.append(z[0][0])

        # make the result list an array
        results = np.array(results)
        return results

    def train_genetic(self, x, y):

        # run a normal iteration as forward process for all samples provided
        # it's important to notice, that we have only one epoch in this case,
        # since we do not use a gradient descent method to update the weights
        # so we check the performance after each forward pass
        predictions = self.predict(x)
        performance = self._calculate_fitness(predictions, y)


        print(f"z with shape: {z.shape}")
        return predictions
