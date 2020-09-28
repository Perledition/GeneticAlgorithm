# import standard modules
import os
import random
from copy import deepcopy

# import from third party modules
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# import project related modules
from Network.sequence import GeneticSequence


class GeneticModel:

    def __init__(self, architecture, fitness_limit: float = 2.00, generation_limit: int = 100,
                 population_size: int = 200):

        # defines the boundaries which will be used in order to define an end of the runtime
        # since the main training runs in a while true loop, the limit conditions are needed in order to
        # avoid an infinity loop
        self.fitness_limit = fitness_limit                # set limit for fitness based on the fitness function
        self.generation_limit = generation_limit          # set a maximum of generations to produce

        self.architecture = architecture                  # NumpyNet architecture to use in order to create a population
        self.population_fitness_avg = list()              # a list to collect the fitness of all members of a population
        self.best_fitness = 0                             # holds the highest fitness value of the current population
        self.worst_fitness = 0                            # holds the worst fitness value of the current population

        # create the population from generate population with the desired size of the population
        self.population = self._generate_population(population_size)

    def _generate_population(self, population_size: int):
        """
        class internal function which is only used as an initial step to create a starting population.
        Each Member of the population is generated from the NumpyNet architecture handed over to the GeneticModel.

        :param population_size: int: defines the size of the population

        :return: list: returns a list with NumpyNet Models
        """

        # count from 0 to population size and and generate a GeneticSequence NumpyNet for each count step
        # Genetic Sequence is a specific wrapper similar to Sequence from NumpyNet. However, GeneticSequence is
        # created in order to fullfill the requirements for the GA. Read more about GeneticSequence in it's class
        # description.
        # each time a model is created, the list of layers (self.architecture) is copied deeply. This step is needed
        # to ensure that each model points at a different location of memory and therefore is an individual.
        population = [GeneticSequence(
            deepcopy(self.architecture),
        ) for _ in range(0, population_size)]

        return population

    def _selection(self):
        """
        class internal function triggering the selection process of the fittest parents from the population.
        The function chooses only two parents for each run form the population. So there is no multi-parent crossover
        supported yet, even if it would be easy to implement.

        :return: tuple: returns a sorted population based on fitness, first parent selected, second parent selected
        """

        # sorts the population based on it's performance values in descending order
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # count unique fitness values
        unique_fitness = len(list(set(x.fitness for x in sorted_pop)))

        # if unique fitness is higher than one pick a parent from the top two groups in order
        # to promote heterogeneous chromosome pool - this is especially helpful, when a bunch of networks share the
        # best performance value.
        if unique_fitness > 2:

            # collect all available fitness values within the population and sort them in descending order.
            # then pick the top to entries from the list, since these values.
            sorted_fitness = sorted(list(set(x.fitness for x in sorted_pop)), reverse=True)
            top_1_fitness = sorted_fitness[0]
            top_2_fitness = sorted_fitness[1]

            # collect parents which full fill the criteria of the highest fitness value and do a identical selection but
            # for the second highest fitness value.
            top_1_parents = [x for x in sorted_pop if x.fitness == top_1_fitness]
            top_2_parents = [x for x in sorted_pop if x.fitness == top_2_fitness]

            # pick a member / parent randomly from the above defined collections for parent 1 and parent 2
            # the random function will return the only parent available in case a collection has just one entry.
            parent1 = random.choice(top_1_parents)
            parent2 = random.choice(top_2_parents)

        else:
            # in case only one fitness value over the entire population is available pick randomly two parents from
            # the sorted population. It doesn't matter which one since the sorting has no meaning, if all parents share
            # the same fitness value.
            parent1 = random.choice(sorted_pop)
            parent2 = random.choice(sorted_pop)

        # set the highest and the lowest fitness value as global attribute of the GeneticModel, since they will
        # be needed at different places again.
        self.best_fitness = sorted_pop[0].fitness
        self.worst_fitness = sorted_pop[-1].fitness

        return sorted_pop, parent1, parent2

    def _crossover(self, p1_weights: np.array, p2_weights: np.array, proba: float):
        """
        class internal function that performance the chromosome crossover between two parents. for each row and column
        combination a value will be selected randomly either from parent 1 or from parent 2. However, a probability
        value defines how likely it is to pick the chromosomes from parent 1. The probability should be based on the
        fitness value of the parents, so that the fitter parent inherits its genes with an increased probability.

        :param p1_weights: numpy.array: chromosomes / weights matrices of parent 1
        :param p2_weights: numpy.array: chromosomes / weights matrices of parent 2
        :param proba: float: defines the likelihood, that parent 1 inherits it's genes

        :return: numpy.array: returns a new child chromosome / weight
        """

        assert p1_weights.shape == p2_weights, "parent1 and parent2 must have the same input in order to make crossover"

        # get the amount of rows and columns for the matrix
        rows, columns = p1_weights.shape

        # initialize an empty child array with zeros. The zeros will be filled with parent values from parent 1 oder
        # parent 2
        child_weight = np.zeros((rows, columns))

        # create a simple helper function which will pick a input value y (value of parent 2) in case a random generated
        # value is below the proba parameter. Otherwise it will take the x value (value from parent 1)
        pick_func = lambda x, y, fitness: y if random.random() < fitness else x

        # for each row and column combination assign a chromosome value based on parents fitness likelihood
        for r in range(0, rows):
            for c in range(0, columns):
                child_weight[r, c] = pick_func(p1_weights[r, c], p2_weights[r, c], proba)

        # apply the mutation function with it's default probability and return the new child chromosome
        child_weight = self._mutation(child_weight)
        return child_weight

    def _mutation(self, weight: np.array, probability: float = 0.2):
        """
        class internal function to do a mutation operation on a given matrix. Mutation is a process where a chromosome
        is modified individually and randomly. This can lead to a better or worse fitness performance, since it's not
        independent from the heritage of the parents. The function does randomly choose and row and column index from
        the matrix and adds or subtracts a random value, if a random value is smaller than a given
        probability. This process is (also randomly) 1 - 3 repeated.

        :param weight: np.array: weight matrix / chromosome to mutate
        :param probability: float: likelihood of applying a mutation.

        :return: returns the input matrix / chromosome either with random modification or without it
        """

        # randomly choose an index for the mutation on row and column level and run the for loop 1 - 3 times
        for _ in range(random.randint(1, 3)):

            # if a random value is lower than the function parameter proba do a mutation
            if random.random() < probability:

                # choose randomly a row and column index and add or subtract a random value to the selected index
                row = random.choice([r for r in range(weight.shape[0])])
                column = random.choice([c for c in range(weight.shape[1])])
                weight[row, column] -= (random.random() * random.choice([-1, 1]))

        # after the for loop finished return the matrix
        return weight

    def _child_generation(self, parents: list, replace_size: int):
        """
        class internal function which creates children of the current population based on the most fittest parents.
        A child has new weights and biases for each densely connected layer. Furthermore a child gets a new hex color
        code id in order to make inheritance more visible. the color code (first two value from parent 1, second two
        from parent 2 und last two newly generated.

        :param parents: list: a list of size two. parent 1 at index 0 and parent 2 at index 1
        :param replace_size: int: max amount of population members to replace with children.

        :return: dict: returns a dictionary structure of children. Key: child_id (hex code), Values: weights and bias
        """

        # define the amount of children to produce - can be one to the maximum amount of replacement handed over as
        # parameter
        amount_children = random.randint(1, replace_size)

        # get the weights, biases and fitness of the first and second parent. get_all_... functions are provided
        # by the GeneticSequence class. (wrapper like Sequence from NumpyNet)
        parent1_weights = parents[0].get_all_weights()
        parent1_bias = parents[0].get_all_bias()
        parent1_fitness = parents[0].fitness

        parent2_weights = parents[1].get_all_weights()
        parent2_bias = parents[1].get_all_bias()
        parent2_fitness = parents[1].fitness

        # define a likelihood to which parent 1 will inherit it's chromosomes
        heritage_proba = parent1_fitness / (parent2_fitness + parent1_fitness)
        print(f"evolve: {parents[0].id} with heritage_proba: {heritage_proba} and"
              f"{parents[1].id} with heritage_proba: {1 - heritage_proba}")

        # Start the main loop of creating new children, with i being a counter of iterations and children being an
        # empty dict which serves to collect all produced children.
        i = 0
        children = dict()

        # run the creation of new children until the counter reached the size of the amount of children to produce
        # which was defined above.
        while i < amount_children:

            # initialize an empty list for child weights and child biases
            child_w, child_b = list(), list()

            # for each densely connected layer a new weight and bias must be defined.
            for layer_index in range(len(parent1_weights)):

                # get the specific weights and biases form parents with the current index of the iteration
                p1_weights = parent1_weights[layer_index]
                p2_weights = parent2_weights[layer_index]
                p1_bias = parent1_bias[layer_index]
                p2_bias = parent2_bias[layer_index]

                # create a child weight and bias with the crossover function and
                # the likelihood to which parent 1 will inherit it's chromosomes
                child_weight = self._crossover(p1_weights, p2_weights, heritage_proba)
                child_bias = self._crossover(p1_bias, p2_bias, heritage_proba)

                # append weights and bias (new chromosomes) to the list / collection
                child_w.append(child_weight)
                child_b.append(child_bias)

            # generate a new child id with is partly form the parents and partly new
            child_id = parents[0].id[:2] + parents[1].id[2:5] + "%02X" % random.randint(0, 255)

            # enter the child with id, weights and bias to the children dict and increase the count
            children[child_id] = {"weights": [deepcopy(wl) for wl in child_w], "bias": [deepcopy(bl) for bl in child_b]}
            i += 1

        # give an information how many children were produced and return the dict
        print("amount children produced: ", len(children))
        return children

    def generation_status(self, generation: int):
        """
        print a quick information about the current population and in which generation the population is.

        :param generation: int: number of generation

        :return: None
        """
        print(f"GENERATION: {generation}")
        print("==========================")
        print(f"Population size: {len(self.population)}")
        print(f"Avg. Fitness: {self.population_fitness_avg[-1]}")
        print(f"Best: {self.best_fitness}")
        print(f"Worst: {self.worst_fitness}")
        print("")

    def generation_plot(self, generation: int):
        """
        function generates a matplotlib graph which displays information about the current generation and saves it
        if the working directory in a new folder run.

        :param generation: int: number of generation

        :return: None
        """

        # define how the spacing of the chart
        plt.figure(figsize=(20, 10))

        # get the current population values about fitness and round them all to the same amount of digits in order
        # to make the displayed numbers consistent
        best_title = round(self.best_fitness, 6)
        worst_title = round(self.worst_fitness, 6)
        avg_title = round(self.population_fitness_avg[-1], 6)

        # add a main title on the graph
        plt.suptitle(f"Generation: {generation}, best fit: {best_title}, worst fit: {worst_title}, "
                     f"avg. fit: {avg_title}", fontsize=16)

        # create fitness scatter plot. Each member of the population will be a single point in the graph displayed with
        # the hex color code color of it's id.
        plt.subplot(1, 3, 1)
        for member in self.population:
            plt.scatter(member.avg_precision, member.avg_recall, color=member.id)

        # define x and y descriptions and axis limits for the scatter plot
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        # bar plot to display fitness divergence
        plt.subplot(1, 3, 2)
        plt.bar(["Best Fitness", "Worst Fitness"], [self.best_fitness, self.worst_fitness])
        plt.ylabel("fitness")
        plt.ylim(0, 2)

        # define a line chart in order to display how the average fitness of the population has changed compared
        # to all previous generations
        plt.subplot(1, 3, 3)
        plt.plot([x for x in range(1, generation + 1)], self.population_fitness_avg)
        plt.xlabel("generation")
        plt.ylabel("avg. population fitness")
        plt.ylim(0, 2)

        # in case the folder "run" is not available in the current working directory yet, create it
        if not os.path.isdir("run"):
            os.mkdir(os.path.join(os.getcwd(), "run"))

        # save the image as png file in the run folder with the current generation id
        plt.savefig(os.path.join(os.getcwd(), f"run/generation{generation}.png"))

    def to_gif(self):
        """
        a function to generate a gif file from all generated and saved plots from in the run folder.

        :return: None
        """
        # get all paths sort the images and generate the gif file.
        file_path_in = os.path.join(os.getcwd(), "run/generation*.png")
        file_path_out = os.path.join(os.getcwd(), "run/experiment.gif")
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(file_path_in))]
        img.save(fp=file_path_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)

    def train(self, x: np.array, y: np.array, generation_replacement: float = 0.2, monitor: bool = False):
        """
        main function of the model. it will run the entire training with given feature and label data in order to create
        a NumpyNet model with is able to run predictions. It executes the main genetic algorithm order of:

            - run prediction and measure fitness
            - select fittest members of population
            - run a cross over in order to generate children from the fittest members of the population
            - add children to the population and remove members with the lowest fitness

        as long if the limit of generations is reached or if the avg value of the population reached the maximum
        of expected fitness.

        :param x: numpy.array: holds and nxm matrix with training data of features
        :param y: numpy.array: holds and nx1 matrix of labels accordingly to feature data x
        :param generation_replacement: float: defines a percentage threshold of maximum replacement of the population
                                              with children
        :param monitor: bool: defines whether the complete training process will be captures and pngs and gif or not
                              (False default)
        :return: GeneticSequence model of best performing (fittest) NumpyNet model
        """

        # convert the parameter generation_replacement from an relative percentage to an absolute value
        # since the amount of members to replace in the population is depended on the population size
        replace_size = int(len(self.population) * generation_replacement)

        # set variables to control the evolution
        generation = 1
        evolve = True

        # run the following evolve process until either the limit of generations is reached or if the avg. fitness
        # limit is archived
        while evolve:

            # run prediction for each member of the population in order to estimate it's fitness
            # then calculate the overall fitness of the population
            for member in self.population:
                member.train_genetic(x, y)

            # calculate the average fitness of the entire population
            self.population_fitness_avg.append(sum([member.fitness for member in self.population]) / len(self.population))

            # select the two most fittest parents and perform a cross over for the weights (chromosomes)
            sorted_parents, parent1, parent2 = self._selection()

            # pick the most fittest parents (parent1 and parent2) and generate children (crossover)
            new_weights = self._child_generation([parent1, parent2], replace_size)

            # with the new weights and biases (chromosomes) for each new child, set a new GeneticSequence Model
            # (Specific version of a NumpyNet Model) and assign the new child weights and biases to it.
            # initialize an empty list to collect all new models (children)
            children = list()

            # for each new child run the process of model creation
            for child_id, child_values in new_weights.items():

                # set initialize a GeneticSequence with a deepcopy of the NumpyNet Layer structure
                # deepcopy is again used in order to ensure, that the new model does point to an individual
                # place within the memory and does not point to the same location as another model.
                child = GeneticSequence(deepcopy(self.architecture))

                # update weights, bias and id with the child specific values and add it to the collection
                child.update_weights(child_values["weights"])
                child.update_bias(child_values["bias"])
                child.id = child_id
                children.append(deepcopy(child))

            # copy the sorted population but without the last n members of it (n is defined by the amount of children)
            sorted_parents = deepcopy(sorted_parents[:-len(children)])

            # add all children to the population and shuffle the population so the population is not any longer sorted
            # in a certain way.
            self.population = deepcopy(sorted_parents + children)
            random.shuffle(self.population)

            # print a status update in the console about the current generation
            self.generation_status(generation)

            # if the monitor argument is true create and save a visual for the current generation
            if monitor:
                self.generation_plot(generation)

            # check if one of the conditions to stop the algorithm is true otherwise count plus one on the generation
            if (generation == self.generation_limit) or (self.best_fitness == self.fitness_limit):
                evolve = False
            else:
                generation += 1

        # return the member with the highest fitness from the population
        print("stopped evolution")
        print(f"avg_recall: {sorted_parents[0].avg_recall}")
        print(f"avg_precision: {sorted_parents[0].avg_precision}")

        # if the monitor argument is true generate a final gif of all plots in order to have a dynamic picture of
        # how the evolution took place
        if monitor:
            self.to_gif()

        # return the fittest model (GeneticSequence) of the current population
        return sorted_parents[0]
