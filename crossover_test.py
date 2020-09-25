import numpy as np
import random


X = np.random.randn(10, 3)
Y = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
w1 = np.array([[-0.06590602, -0.04169132],
               [0.06032958, -0.07557409],
               [-0.00455764, 0.00846962]])

w2 = np.array([[0.15664711],
               [-0.05540206]])

print(X[0].reshape(-1, 1).shape)
d1 = np.dot(X[0].reshape(-1, 1).T, w1)
print(np.dot(d1, w2))

w12 = np.array([[0.01336059, -0.09232362],
                [0.01336059, -0.09232362],
                [0.90961304, -0.00316533]])

w22 = np.array([[0.09419786],
                [0.09419786]])

d1 = np.dot(X[0].reshape(-1, 1).T, w12)
print(np.dot(d1, w22))