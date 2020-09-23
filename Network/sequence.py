# import standard modules
import inspect

# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# import project related modules
from Network.dense import Dense


class AUC:

    def _false_positive_count(self):
        pass

    def _true_positive_count(self):
        pass

# TODO: USE AUC as Fitness function since it provides the max character and allows easy calculation
class GeneticSequence(object):

    def __init__(self, layer: list, model_type="classification"):
        self.layer = layer
        self.model_type = {"classification": 0, "regression": 1}

    def get_all_weights(self):
        return [l.weights for l in self.layer if isinstance(l, Dense)]

    def update_weights(self, weights):
        pass

    def calculate_fitness

    def train_genetic(self, x, y):

        # run a normal iteration as forward process for all samples provided
        # it's important to notice, that we have only one epoch in this case,
        # since we do not use a gradient descent method to update the weights
        # so we check the performance after each forward pass
        for sample in x:

            for layer in self.layer:
                z = layer.forward(x)

            # the loss is not needed in this case
            self.performance.add()
