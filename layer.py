import numpy as np


class Layer:
    def __init__(self, *, W, B, activation_function):
        self._W = W
        self._B = B
        self._activation_function = activation_function

    def transmit(self, X):
        temp_A = np.dot(X, self._W) + self._B
        return self._activation_function(temp_A)

