# import standard modules
from collections import namedtuple
from random import choices, randint, randrange, randrange, random

# import from third party modules
import numpy as np
import sklearn.datasets as skd

# import project related modules
from Network.sequence import GeneticSequence
from Network.dense import Dense
from Network.activation import Sigmoid

# idea: each member of a population is a byte array which is created from the weights of a network, it's fitness is
# defined by it's loss accuracy and it's loss value


# construct which holds the values for one network
class Member:

    def __init__(self, model, measure_functions):
        self.model = model
        self.weights = model.get_all_weights()
        self.fitness = self._member_fitness(measure_functions)
        self.genes = self._convert_weights()

    def _convert_weights(self):
        genes = np.concatenate(tuple(x.ravel() for x in self.weights)).reshape(-1, 1)
        return genes

    def _member_fitness(self, measures):
        return 0


class GeneticModel:

    def __init__(self, architecture, fitness_limit=1.00, generation_limit=100, population_size=10):

        # defines the boundaries which will be used in order to define an end of the runtime
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit

        # defines the population which will be used in order to evolve the algorithm
        # is a list of NumpyNet models defined outside of this class
        self.population = self._generate_population(architecture, population_size)

    def _generate_population(self, architecture, population_size):

        # generate a bunch of models with the same architecture but random weights
        population = [GeneticSequence(
            architecture,
        ) for _ in range(0, population_size)]

        return population

    def _sort_population(self):
        pass

    def _selection(self):
        pass

    def _crossover(self):
        pass

    def _mutation(self):
        pass

    def get_population_fitness(self):
        pass

    def train(self, population_size=10):

        # assign population to global population value
        return 0


x = np.random.randn(10, 3)
y = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
print(x.shape)
layers = [
    Dense(2, 3, input_layer=True),
    Sigmoid(),
    Dense(1, 2, activation="softmax")
]


model = GeneticSequence(
    layers,
)

print(model.train_genetic(x, y))
