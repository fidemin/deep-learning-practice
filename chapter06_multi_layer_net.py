import matplotlib.pyplot as plt
import numpy as np

from core.layernet import MultiLayerNet, ActivationType
from core.parameter_updaters import SGD, Momentum, AdaGrad
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

num_of_iters = 6000
train_size = x_train.shape[0]
input_size = x_train.shape[1]
output_size = t_train.shape[1]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
print('train size:', train_size, 'input size:', input_size, 'iter_per_epoch:', iter_per_epoch)

network = MultiLayerNet(
    input_size=input_size, hidden_size_list=[50], output_size=output_size,
    use_he_init=True, activation=ActivationType.Relu)
# parameter_updater = SGD(learning_rate=0.01)
# parameter_updater = Momentum()
parameter_updater = AdaGrad()

for i in range(num_of_iters):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)
    parameter_updater.update(network.params, grad)

    # records results
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print('train acc, test acc | ' + str(train_acc) + ", " + str(test_acc))


# 그래프 그리기
plt.plot(np.arange(len(train_acc_list)), train_acc_list, label='train accuracy')
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label='test accuracy', linestyle='--')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
