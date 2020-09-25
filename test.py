# import standard modules
import random
from copy import deepcopy
from collections import namedtuple


# import from third party modules
import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets as skd

# import project related modules
from Network.sequence import GeneticSequence
from Network.dense import Dense
from Network.activation import Sigmoid

# idea: each member of a population is a byte array which is created from the weights of a network, it's fitness is
# defined by it's loss accuracy and it's loss value


class GeneticModel:

    def __init__(self, architecture, fitness_limit=1.00, generation_limit=100,
                 population_size=200, population_size_max=1000):

        # defines the boundaries which will be used in order to define an end of the runtime
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit
        self.architecture = architecture

        # defines the population which will be used in order to evolve the algorithm
        # is a list of NumpyNet models defined outside of this class
        self.population_size_max = population_size_max
        self.population = self._generate_population(population_size)
        self.population_fitness_avg = 0

    def _generate_population(self, population_size):

        # generate a bunch of models with the same architecture but random weights
        population = [GeneticSequence(
            self.architecture,
        ) for _ in range(0, population_size)]

        return population

    def _selection(self):

        # sorts the population based on it's performance values in descending order
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # return the sorted population as well as the fitness values for the best and the worst performing member
        # of the population
        return sorted_pop, sorted_pop[0].fitness, sorted_pop[-1].fitness

    def _crossover(self, parents, chromosome_switch=10):

        print(f"evolve: {parents[0].id} and {parents[1].id}")
        parent1_weights = parents[0].get_all_weights()
        parent1_weights_new = list()

        parent2_weights = parents[1].get_all_weights()
        parent2_weights_new = list()

        assert len(parent1_weights) == len(parent2_weights), "layer structure of networks must be equal for crossover"

        for layer_index in range(len(parent1_weights)):

            # get the specific weights
            p1_weights = parent1_weights[layer_index]
            p2_weights = parent2_weights[layer_index]

            i = 0
            while i < chromosome_switch:
                # get the size of rows and columns available
                columns = p2_weights.shape[1]
                rows = p2_weights.shape[0]

                # get a random cut point
                c = random.randint(0, columns-1)
                r = random.randint(0, rows-1)

                # exchange weights at random row pick
                p1_weights[r, c] = p2_weights[r, c]
                p2_weights[r, c] = p1_weights[r, c]

                i += 1

            p1_weights = self._mutation(p1_weights)
            p2_weights = self._mutation(p2_weights)

            parent1_weights_new.append(p1_weights)
            parent2_weights_new.append(p2_weights)

        # print(parent1_weights_new)
        # A9A78A
        child1_id = parents[0].id[:2] + parents[1].id[2:4] + "%02X" % random.randint(0, 255)
        child2_id = parents[1].id[:2] + parents[0].id[2:4] + "%02X" % random.randint(0, 255)
        new_weights = {
            child1_id: deepcopy(parent1_weights_new),
            child2_id: deepcopy(parent2_weights_new)
        }
        return new_weights

    def _mutation(self, weight, probability: float = 0.2):

        # randomly choose an index for the mutation on row and column level
        row = random.choice([r for r in range(weight.shape[0])])
        column = random.choice([c for c in range(weight.shape[1])])

        if random.random() < probability:
            weight[row, column] = abs(weight[row, column] - 1)

        return weight

    def generation_status(self, generation, best, worst):
        print(f"GENERATION: {generation}")
        print("==========================")
        print(f"Population size: {len(self.population)}")
        print(f"Avg. Fitness: {self.population_fitness_avg}")
        print(f"Best: {best}")
        print(f"Worst: {worst}")
        print("")

    def generation_plot(self, generation, best, worst):
        plt.figure(figsize=(10, 10))
        plt.title(f"GA: {generation}, best fit: {best}, worst fit: {worst}, avg. fit: {self.population_fitness_avg}")
        for member in self.population:
            plt.scatter(member.avg_precision, member.avg_recall, color=member.id, label=member.id)

        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.legend()
        plt.show()

    def train(self, x, y, generation_replacement=0.2):
        starting_pop = [x.id for x in self.population]
        replace_size = int(len(self.population) * generation_replacement)
        # set variables to control the evolution
        generation = 1
        evolve = True

        while evolve:
            # run prediction on each member of the population in order to estimate it's fitness
            # then calculate the overall fitness of the population
            for member in self.population:
                member.train_genetic(x, y)

            # calculate the average fitness of the entire population
            self.population_fitness_avg = sum([member.fitness for member in self.population]) / len(self.population)

            # select the two most fittest parents and perform a cross over for the weights (chromosomes)
            sorted_parents, best_fitness, worst_fitness = self._selection()

            # pick the most fittest parents - so the first two entries since the array is sorted in descending order
            new_weights = self._crossover(sorted_parents[:2])

            # generate new members of the population and get rid of the worst performing members
            # get all members which have an equal bad fitness
            if (generation == 1) or (generation == self.generation_limit):
                self.generation_plot(generation, best_fitness, worst_fitness)

            dinos = [dino for dino in self.population if dino.fitness == worst_fitness]
            if len(dinos) > 2:
                print("random pick")
                random2_dinos = random.choices(dinos, k=replace_size)

                for d in random2_dinos:
                    try:
                        self.population.remove(d)
                    except ValueError:
                        continue

            else:
                print("slice")
                self.population = deepcopy(sorted_parents[:-replace_size])

            ix = 0
            childs = list()
            while ix < replace_size:
                for child_id, child_weights in new_weights.items():
                    child = GeneticSequence(self.architecture)
                    child.update_weights(child_weights)
                    child.id = child_id
                    childs.append(child)
                    ix += 1

            if len(self.population) + len(childs) > self.population_size_max:
                additional_removal = (len(self.population) + len(childs)) - self.population_size_max
                self.population = deepcopy(self.population[:additional_removal])

            self.population = deepcopy(self.population + childs)

            # print a status update in the console about the current generation
            self.generation_status(generation, best_fitness, worst_fitness)

            if generation == self.generation_limit:
                evolve = False
            else:
                generation += 1

        # return the member with the highest fitness from the population
        print("stopped evolution")
        print(f"avg_recall: {sorted_parents[0].avg_recall}")
        print(f"avg_precision: {sorted_parents[0].avg_precision}")

        final_pop = [x.id for x in self.population]
        print(starting_pop)
        print(final_pop)
        return sorted_parents[0]


X = np.random.randn(10, 3)
Y = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0])

layers = [
    Dense(2, 3, input_layer=True),
    Sigmoid(),
    Dense(3, 2),
    Sigmoid(),
    Dense(2, 3, activation="softmax")
]

GeneticModel(
    layers,
    generation_limit=1000,
    population_size=50,
    population_size_max=1000,
).train(X, Y, generation_replacement=0.2)