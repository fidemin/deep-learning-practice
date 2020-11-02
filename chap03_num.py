import sys, os
from dataset.mnist import load_mnist
import pickle

from layer import Layer
import numpy as np
from activation_function import sigmoid, identity

def load_data():
    # 10000개의 데이터 셋을 가져온다.
    return load_mnist(flatten=True, normalize=True)


def load_weight():
    with open('sample_data/sample_weight.pkl', 'rb') as f:
        weight_dict = pickle.load(f)

    return weight_dict


if __name__ == '__main__':
    weight = load_weight()
    layer_0 = Layer(W=weight['W1'], B=weight['b1'], activation_function=sigmoid)
    layer_1 = Layer(W=weight['W2'], B=weight['b2'], activation_function=sigmoid)
    layer_2 = Layer(W=weight['W3'], B=weight['b3'], activation_function=identity)

    # x_test는 (10000, 784) size의 행렬이다. 총 10000개의 데이터셋이 있다.
    (_, _), (x_test, t_test) = load_data() 

    # Y는 (10000, 10) size의 행렬이다. 10개 중 가장 큰 값이 그 데이터의 예상되는 숫자이다.
    Y = layer_2.transmit(
        layer_1.transmit(
            layer_0.transmit(x_test)))

    accuracy_cnt = 0
    for i, expected_number in enumerate(t_test):
        p_number = np.argmax(Y[i])
        if p_number == expected_number:
            accuracy_cnt += 1

    print("Accuracy: ", float(accuracy_cnt) / len(x_test))

