# import standard modules
import random
from copy import deepcopy

# import third party modules
import numpy as np

# import project related modules
from simple_numpy_net.dense import Dense
from simple_numpy_net.metric import NaiveClassificationMatrix


class GeneticSequence(object):
    """
    GeneticSequence is a wrapper class like the original Sequence class from NumpyNet. However, GeneticSequence is
    specifically designed to meet the requirements of a Genetic Algorithm Model. It will be used by GeneticModel in
    order to create Genetic Algorithm specific models in the population. It's main difference to the NumpyNet
    Sequence is the ability to update weights, and it's fitness value orientation. It is not trainable by it self
    like the Sequence class from NumpyNet. Nevertheless it can be used to do predictions on unseen data.

    :param: layers: list: layer architecture - list of layers to execute

    """

    def __init__(self, layers: list):
        self.id = self.generate_color_id()                    # hex color code which serves for identification of roots
        self.layers = layers                                  # layer structure of the model
        self.fitness = 0                                      # models fitness
        self.avg_precision = 0                                # models avg precision performance over all classes
        self.avg_recall = 0                                   # models avg recall performance over all classes
        self.fitness_func = self._classification_fitness      # fitness function of the model / (classifi. only for now)

    @staticmethod
    def generate_color_id():
        """
        static function to generate a random hex color code

        :return: str: hex color code
        """
        # helper function to create a random digit between 0 and 255. then applied three times to a hex string.
        r = lambda: random.randint(0, 255)
        return '#%02X%02X%02X' % (r(), r(), r())

    def get_all_weights(self):
        """
        function does extract all weights from densely connected layers of the layer structure and collect the weight
        matrices (chromosomes) in a list.

        :return: list: with weight matrices
        """

        # add weights for each layer if layer is a Dense layer and return the list
        return [l.weights for l in self.layers if isinstance(l, Dense)]

    def get_all_bias(self):
        """
        function does extract all biases from densely connected layers of the layer structure and collect the biases
        matrices (chromosomes) in a list.

        :return: list: with biases matrices
        """

        # add bias for each layer if layer is a Dense layer and return the list
        return [b.bias for b in self.layers if isinstance(b, Dense)]

    def update_weights(self, new_weights: list):
        """
        function allows to exchange weights (chromosomes) of a layer structure with other or new ones.
        this especially helpful in the context of GA training, when crossover weights must be assigned.
        make sure that the amount of weights are the same as the amount of densely connected layer the the layer
        structure of the model.

        :param new_weights: list: of numpy.arrays with weights (chromosomes)

        :return: None
        """

        # perform a quick quality check in order to ensure each layer gets new weights and out of bounds
        # errors are avoided
        d_layers = len([1 for _ in self.layers if isinstance(_, Dense)])
        assert len(new_weights) == d_layers, "amount of new weights does not fit the count of Dense Layer in the Model"

        # set a count in order to keep track of the right list index
        crnt_index = 0

        # iterate over each layer of the model and check if the layer is a Dense Layer
        for layer in range(len(self.layers)):
            if isinstance(self.layers[layer], Dense):

                # if the current layer is a Dense layer assign new weights at the current list index and count index + 1
                self.layers[layer].weights = deepcopy(new_weights[crnt_index])
                crnt_index += 1

    def update_bias(self, new_bias: list):
        """
        function allows to exchange the biases (chromosomes) of a layer structure with other or new ones.
        this especially helpful in the context of GA training, when crossover biases must be assigned.
        make sure that the amount of biases are the same as the amount of densely connected layer the the layer
        structure of the model.

        :param new_bias: list: of numpy.arrays with biases (chromosomes)

        :return: None
        """

        # perform a quick quality check in order to ensure each layer gets new biases and out of bounds
        # errors are avoided
        d_layers = len([1 for _ in self.layers if isinstance(_, Dense)])
        assert len(new_bias) == d_layers, "amount of new weights does not fit the count of Dense Layer in the Model"

        # set a count in order to keep track of the right list index
        crnt_index = 0

        # iterate over each layer of the model and check if the layer is a Dense Layer
        for layer in range(len(self.layers)):
            if isinstance(self.layers[layer], Dense):

                # if the current layer is a Dense layer assign new weights at the current list index and count index + 1
                self.layers[layer].bias = new_bias[crnt_index]
                crnt_index += 1

    def _classification_fitness(self, y_hat: np.array, y: np.array):
        """
        class internal function which calculates the overall fitness of the model. The fitness for a classification
        problem is defined be the average of precision and recall. Concrete:

            fitness = avg. precision + avg. recall.

        this simple yet effective fitness function has a clear maximization character and offer at the same moment
        a simple to define maximum. ones precision or recall over all classes is 1.0, the model makes perfect
        predictions. The absolute best fitness value would be 2.0 in consequence. However, it's recommended to set the
        limit of fitness to somewhat lower than 2, since the model is likely to be heavily over fitted, once it
        has a fitness of 2.0. There might be other (better) fitness functions but for this experiment it turned out to
        be effective.

        :param y_hat: numpy.array: of prediction data
        :param y: numpy.array: of label data
        :return: float: fitness value of the model
        """
        # get a classification matrix in order to get an overview of predictions and it's true values
        # the NaiveClassificationMatrix is a simplified version of the ClassificationMatrix class from NumpyNet
        matrix = NaiveClassificationMatrix().fit(y_hat, y)

        # initialize lists for precision and recall to collect the values over all classes
        precision_count = list()
        recall_count = list()

        # calculate precision and recall for each class
        for target_class in range(matrix.shape[0]):

            # get count values for each class from the classification matrix
            tp = matrix[target_class, target_class]
            fn = np.sum(np.take(matrix[target_class], [ix for ix in range(matrix.shape[0]) if ix != target_class]))
            fp = np.sum(matrix[:, target_class]) - tp

            # calculate precision and recall and collect the results in the list
            if tp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            precision_count.append(precision)

            if tp > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            recall_count.append(recall)

        # calculate the average precision and recall performance
        self.avg_precision = sum(precision_count) / len(precision_count)
        self.avg_recall = sum(recall_count) / len(recall_count)

        # calculate the fitness of the model by adding together avg. recall and avg. precision
        fitness = self.avg_precision + self.avg_recall
        return fitness

    def predict(self, x: np.array):
        """
        function to predict a batch of feature samples.

        :param x: numpy.array: matrix of shape nxm with n samples and m features (features must fit layer structure)

        :return: np.array: resulting classification values
        """
        # list for collecting the result of each sample
        results = list()

        # run the prediction process for each sample in x
        for ix in range(0, x.shape[0]):

            # forward feed through all layers to get a final prediction
            z = x[ix].reshape(-1, 1)
            for i, layer in enumerate(self.layers):
                z = self.layers[i].forward(z)

            # append results to list of results - the resulting class is the index of the maximum probability
            # from the one hot vector
            results.append(np.argmax(z))

        # make the result list an numpy array and return it
        return np.array(results)

    def train_genetic(self, x: np.array, y: np.array):
        """
        main function of GeneticSequence model. it will run the prediction for x and calculates based on the prediction
        results the fitness of the model.

        :param x: numpy.array: feature matrix of shape nxm, with m fitting the layer structure input dimensions
        :param y: numpy.array: matrix / vector of labels accordingly to feature matrix x

        :return: None
        """

        # run a normal iteration as forward process for all samples provided
        # it's important to notice, that we have only one epoch in this case,
        # since we do not use a gradient descent method to update the weights
        # so we check the performance after each forward pass
        predictions = self.predict(x)
        self.fitness = self._classification_fitness(predictions, y)
