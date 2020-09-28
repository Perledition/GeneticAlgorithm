# import standard modules
import os
import random
import shutil
from copy import deepcopy
from collections import namedtuple


# import from third party modules
import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt
import sklearn.datasets as skd

# import project related modules
from Network.sequence import GeneticSequence
from Network.dense import Dense
from Network.activation import Sigmoid, RelU

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
        self.population_fitness_avg = list()

    def _generate_population(self, population_size):

        # generate a bunch of models with the same architecture but random weights
        population = [GeneticSequence(
            self.architecture,
        ) for _ in range(0, population_size)]

        return population

    def _selection(self):

        # sorts the population based on it's performance values in descending order
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # count unique fitness values
        unique_fitness = len(list(set(x.fitness for x in sorted_pop)))

        # unique fitness is higher than one pick a parent from the top two groups in order
        # to promote heterogeneous chromosome pool
        if unique_fitness > 2:

            # get and sort fitness values
            sorted_fitness = sorted(list(set(x.fitness for x in sorted_pop)), reverse=True)
            top_1_fitness = sorted_fitness[0]
            top_2_fitness = sorted_fitness[1]

            top_1_parents = [x for x in sorted_pop if x.fitness == top_1_fitness]
            top_2_parents = [x for x in sorted_pop if x.fitness == top_2_fitness]

            parent1 = random.choice(top_1_parents)
            parent2 = random.choice(top_2_parents)

        else:
            parent1 = random.choice(sorted_pop)
            parent2 = random.choice(sorted_pop)

        return sorted_pop, parent1, parent2, sorted_pop[0].fitness, sorted_pop[-1].fitness

    def _crossover_weights(self, p1_weights, p2_weights, pick_func, proba: float):
        rows, columns = p1_weights.shape

        # initialize an empty child array
        child_weight = np.zeros((rows, columns))

        # for each row and column assign a chromosome value based on parents fitness likelihood
        for r in range(0, rows):
            for c in range(0, columns):
                child_weight[r, c] = pick_func(p1_weights[r, c], p2_weights[r, c], proba)

        child_weight = self._mutation(child_weight)
        return child_weight

    def _fitness_based_crossover(self, parents, replace_size):

        amount_children = random.randint(1, replace_size)
        parent1_weights = parents[0].get_all_weights()
        parent1_bias = parents[0].get_all_bias()
        parent1_fitness = parents[0].fitness

        parent2_weights = parents[1].get_all_weights()
        parent2_bias = parents[1].get_all_bias()
        parent2_fitness = parents[1].fitness

        heritage_proba = parent1_fitness / (parent2_fitness + parent1_fitness)
        print(f"evolve: {parents[0].id} with heritage_proba: {heritage_proba} and {parents[1].id} with heritage_proba: {1 - heritage_proba}")

        assert len(parent1_weights) == len(parent2_weights), "layer structure of networks must be equal for crossover"
        i = 0
        children = dict()
        while i < amount_children:
            child_w, child_b = list(), list()
            for layer_index in range(len(parent1_weights)):

                # get the specific weights
                p1_weights = parent1_weights[layer_index]
                p2_weights = parent2_weights[layer_index]
                p1_bias = parent1_bias[layer_index]
                p2_bias = parent2_bias[layer_index]

                chromosome_pick = lambda x, y, fitness: y if random.random() < fitness else x
                child_weight = self._crossover_weights(p1_weights, p2_weights, chromosome_pick, heritage_proba)
                child_bias = chromosome_pick(p1_bias, p2_bias, heritage_proba)
                child_bias = self._mutation(child_bias)

                child_w.append(child_weight)
                child_b.append(child_bias)

            child_id = parents[0].id[:2] + parents[1].id[2:5] + "%02X" % random.randint(0, 255)

            children[child_id] = {"weights": [deepcopy(wl) for wl in child_w], "bias": [deepcopy(bl) for bl in child_b]}
            i += 1

        print("amount children produced: ", len(children))
        return children

    def _mutation(self, weight, probability: float = 0.6):

        # randomly choose an index for the mutation on row and column level
        for _ in range(random.randint(1, 3)):
            if random.random() < probability:
                row = random.choice([r for r in range(weight.shape[0])])
                column = random.choice([c for c in range(weight.shape[1])])
                weight[row, column] -= (random.random() * random.choice([-1, 1]))

        return weight

    def generation_status(self, generation, best, worst):
        print(f"GENERATION: {generation}")
        print("==========================")
        print(f"Population size: {len(self.population)}")
        print(f"Avg. Fitness: {self.population_fitness_avg[-1]}")
        print(f"Best: {best}")
        print(f"Worst: {worst}")
        print("")

    def generation_plot(self, generation, best, worst):
        plt.figure(figsize=(20, 10))

        # create fitness scatter plot
        plt.subplot(1, 3, 1)
        for member in self.population:
            plt.scatter(member.avg_precision, member.avg_recall, color=member.id)

        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        # bar plot to display fitness divergence
        plt.subplot(1, 3, 2)
        plt.bar(["Best Fitness", "Worst Fitness"], [best, worst])
        plt.ylabel("fitness")
        plt.ylim(0, 2)

        plt.subplot(1, 3, 3)
        plt.plot([x for x in range(1, generation + 1)], self.population_fitness_avg)
        plt.xlabel("generation")
        plt.ylabel("avg. population fitness")
        plt.ylim(0, 2)

        plt.tight_layout()

        if not os.path.isdir("run"):
            os.mkdir(os.path.join(os.getcwd(), "run"))

        plt.title(f"GA: {generation}, best fit: {best}, worst fit: {worst}, avg. fit: {self.population_fitness_avg[-1]}")
        plt.savefig(os.path.join(os.getcwd(), f"run/generation{generation}.png"))

    def to_gif(self):
        file_path_in = os.path.join(os.getcwd(), "run/generation*.png")
        file_path_out = os.path.join(os.getcwd(), "run/experiment.gif")
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(file_path_in))]
        img.save(fp=file_path_out, format='GIF', append_images=imgs,
                 save_all=True, duration=100, loop=0)

    def train(self, x, y, generation_replacement=0.2, monitor=False):
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
            self.population_fitness_avg.append(sum([member.fitness for member in self.population]) / len(self.population))

            # select the two most fittest parents and perform a cross over for the weights (chromosomes)
            sorted_parents, parent1, parent2, best_fitness, worst_fitness = self._selection()

            # pick the most fittest parents - so the first two entries since the array is sorted in descending order
            new_weights = self._fitness_based_crossover([parent1, parent2], replace_size)

            # generate new members of the population and get rid of the worst performing members
            # get all members which have an equal bad fitness
            # generate children which will be added to the population
            ix = 0
            children = list()
            for child_id, child_values in new_weights.items():
                child = GeneticSequence(self.architecture)
                child.update_weights(child_values["weights"])
                child.update_bias(child_values["bias"])
                child.id = child_id
                children.append(child)
                ix += 1

            # check remove the individuals with the worst fitness
            sorted_parents = deepcopy(sorted_parents[:-len(children)])
            self.population = deepcopy(sorted_parents + children)
            random.shuffle(self.population)
            # print a status update in the console about the current generation
            self.generation_status(generation, best_fitness, worst_fitness)

            if monitor:
                self.generation_plot(generation, best_fitness, worst_fitness)

            if generation == self.generation_limit:
                evolve = False
            else:
                generation += 1

        # return the member with the highest fitness from the population
        print("stopped evolution")
        print(f"avg_recall: {sorted_parents[0].avg_recall}")
        print(f"avg_precision: {sorted_parents[0].avg_precision}")
        if monitor:
            self.to_gif()
        return sorted_parents[0]


def load_extra_datasets():
    N = 300
    gq = skd.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=N, n_features=3, n_classes=2,  shuffle=True, random_state=None)
    return gq


gaussian_quantiles= load_extra_datasets()
X, Y = gaussian_quantiles
X, Y = X, Y
# Y = np.append(Y, np.where(Y==1, 0, 1), axis=1)
print(X.shape, Y.shape)

layers = [
    Dense(4, input_layer=True),
    RelU(),
    Dense(16),
    Sigmoid(),
    Dense(2, activation="softmax")
]

GeneticModel(
    layers,
    generation_limit=150,
    population_size=100,
).train(X, Y, generation_replacement=0.2, monitor=True)