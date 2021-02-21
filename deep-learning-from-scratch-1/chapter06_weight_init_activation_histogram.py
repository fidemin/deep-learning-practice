import matplotlib.pyplot as plt
import numpy as np

from core.activation import sigmoid, relu

x = np.random.randn(1000, 100)
num_of_nodes = 100
number_of_hidden_layer = 5
activations = {}

for i in range(number_of_hidden_layer):
    if i != 0:
        # 첫 히든 레이어가 아니면, 이전 레이어의 활성화 값을 입력 값으로 가져온다.
        x = activations[i-1]

    # standard_deviation = 1
    # standard_deviation = 0.01

    # xavier 초기값
    # standard_deviation = 1 / np.sqrt(num_of_node)

    # He 초기값
    standard_deviation = 1 / np.sqrt(num_of_nodes / 2)
    w = np.random.randn(num_of_nodes, num_of_nodes) * standard_deviation

    a = np.dot(x, w)

    # z = sigmoid(a)
    z = relu(a)

    activations[i] = z


# 히스토그램
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(f'{i+1}-layer')
    plt.hist(a.flatten(), bins=30, range=(0, 1))
    plt.ylim([0, 7000])
plt.show()
