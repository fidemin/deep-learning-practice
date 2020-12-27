import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from core.layernet import TwoLayersNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

num_of_iters = 1000
train_size = x_train.shape[0]
input_size = x_train.shape[1]
print('input_size:', input_size)
batch_size = 100
learning_rate = 0.1
network = TwoLayersNet(input_size=input_size, hidden_size=50, output_size=10)

for i in range(num_of_iters):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # records results
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label='train loss')
plt.xlabel("loss")
plt.ylabel("iterations")
plt.ylim(0, 5.0)
plt.legend(loc='lower right')
plt.show()

