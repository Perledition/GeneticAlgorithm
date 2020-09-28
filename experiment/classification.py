# import standard modules

# import from third party modules
import sklearn.datasets as skd

# import project related modules
from simple_numpy_net.dense import Dense
from simple_numpy_net.activation import RelU, Sigmoid
from simple_numpy_net.metric import NaiveClassificationMatrix
from genetic_extension.models import GeneticModel

###############################
#         set up data         #
###############################


# simple function to create a classification data set - same as in NumpyNet
def generate_data():
    N = 700
    gq = skd.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=N, n_features=3, n_classes=2,  shuffle=True, random_state=None)
    return gq


# generate data
X, Y = generate_data()


###############################
#        set up model         #
###############################

# initialize a GeneticModel (model wrapper for GeneticNumpyNet) with a similar layer structure as in the simple test
# case of NumpyNet in order to compare both models afterwards.
model = GeneticModel([
    Dense(4, input_layer=True),
    RelU(),
    Dense(16),
    Sigmoid(),
    Dense(2, activation="softmax")
],
    generation_limit=500,   # set generation limit to 500
    population_size=100,    # set population size to 100
    fitness_limit=1.9       # make the maximum fitness 1.9
)

# train the model on the first 600 samples from the data
# set monitor to true if you want to generate a gif of your training process
model = model.train(X[:600], Y[:600], generation_replacement=0.2, monitor=False)


###############################
#         test  model         #
###############################
print("\nTestResults")

# make predictions on the last 100 samples from the data set which - not considered in training
# and create a classification matrix in order to see performance on unseen data
test_prediction = model.predict(X[600:])
matrix = NaiveClassificationMatrix().fit(test_prediction, Y[600:])
print(matrix)