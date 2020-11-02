
import numpy as np

from layer import Layer
from activation_function import sigmoid, identity

if __name__ == '__main__':
    layer_0 = Layer(
        W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        B = np.array([0.1, 0.2, 0.3]),
        activation_function = sigmoid)
    layer_1 = Layer(
        W = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        B = np.array([0.1, 0.2]),
        activation_function = sigmoid)
    layer_2 = Layer(
        W = np.array([[0.1, 0.3], [0.2, 0.4]]),
        B = np.array([0.1, 0.2]),
        activation_function = identity)

    X = np.array([1.0, 0.5])
    mid_1_output = layer_0.transmit(X)
    mid_2_output = layer_1.transmit(mid_1_output)
    final_output = layer_2.transmit(mid_2_output)
    print(final_output)
