# import standard modules
import inspect


# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# import project related modules
from Network.dense import Dense
from Network.metric import ClassificationMatrix


# TODO: USE AUC as Fitness function since it provides the max character and allows easy calculation
class GeneticSequence(object):

    def __init__(self, layers: list, model_type="classification"):
        self.layers = layers
        self.performance = 0
        self.fitness_func = {
            "classification": self._classification_fitness,
            "regression": 1
        }[model_type]

    def get_all_weights(self):
        return [l.weights for l in self.layers if isinstance(l, Dense)]

    def update_weights(self, weights):

        for layer in self.layers:
            if layer

    def _classification_fitness(self, y_hat, y):
        """
        is based on precision and recall in average over all classes

        :param y_hat:
        :param y:
        :return:
        """
        # get a classification matrix in order to get an overview of predictions and it's true values
        matrix = ClassificationMatrix().fit(y_hat, y)

        # calculate precision and recall for each class
        precision_count = list()
        recall_count = list()

        for target_class in range(matrix.shape[0]):

            # get count values for each class from the classification matrix
            tp = matrix[target_class, target_class]
            fn = np.sum(np.take(matrix[target_class], [ix for ix in range(matrix.shape[0]) if ix != target_class]))
            fp = np.sum(matrix[:, target_class]) - tp

            # calculate precision and recall and collect the results for each class

            if (tp > 0) and (fp > 0):
                precision = tp / (tp + fp)
            else:
                precision = 0
            precision_count.append(precision)

            if (tp > 0) and (fn > 0):
                recall = tp / (tp + fn)
            else:
                recall = 0
            recall_count.append(recall)

        # calculate the average precision and recall performance
        avg_precision = sum(precision_count) / len(precision_count)
        avg_recall = sum(recall_count) / len(recall_count)

        # the fitness for the classification problem is defined by the area of a rectangle
        # for more information check the general documentation
        fitness = avg_precision * avg_recall
        return fitness

    def _calculate_fitness(self, y_hat, y):
        fitness = self.fitness_func(y_hat, y)
        return fitness

    def predict(self, x):

        results = list()
        for ix in range(0, x.shape[0]):

            # forward feed
            z = x[ix].reshape(-1, 1)
            for i, layer in enumerate(self.layers):
                z = self.layers[i].forward(z)

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
        self.performance = self._calculate_fitness(predictions, y)
        print("performance: ", self.performance)
