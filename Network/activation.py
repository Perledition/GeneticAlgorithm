# import standard modules

# import third party modules
import numpy as np

# import project related modules


class Sigmoid:

    def __init__(self):
        self.cache = None

    def __str__(self):
        return f"Activation: Sigmoid"

    def forward(self, x):
        # line of code to prevent overflow in numpy since this is a common issue
        signal = np.clip(x, -150000, 150000)

        self.cache = 1.0 /(1.0 + np.exp(-1.0 * signal))
        return self.cache

    def backward(self, z):
        return (self.cache * (1.0 - self.cache)) * z


class RelU:

    def __init__(self):
        self.cache = None

    def __str__(self):
        return "Activation: RelU"

    def forward(self, x):
        self.cache = x.copy()
        return np.maximum(0, x)

    def backward(self, z):
        return np.where(self.cache < 0, 0, 1) * z


class Softmax:

    def forward(self, x):
        x = np.clip(x, -150000, 150000)
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)