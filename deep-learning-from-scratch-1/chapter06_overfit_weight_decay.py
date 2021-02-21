import matplotlib.pyplot as plt
import numpy as np

from core.layernet import MultiLayerNet, ActivationType
from core.parameter_updaters import SGD, AdaGrad
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = x_train[:300]
t_train = t_train[:300]
input_size = x_train.shape[1]
output_size = t_train.shape[1]

network = MultiLayerNet(
    input_size=input_size, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=output_size,
    use_he_init=True, activation=ActivationType.Relu, weight_decay_lambda=0.1)

updater = SGD(learning_rate=0.01)

train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0
max_epochs = 201

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    updater.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        epoch_cnt += 1
        print(f'epoch {epoch_cnt}: train acc, test acc | ' + str(train_acc) + ", " + str(test_acc))
        if epoch_cnt >= max_epochs:
            break


plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(np.arange(len(train_acc_list)), train_acc_list, label='train')
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label='train', linestyle='--')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

