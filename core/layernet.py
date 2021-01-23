from collections import OrderedDict

import numpy as np

from core.activation import sigmoid, softmax
from core.gradient import numerical_gradient
from core.layers import AffineLayer, ReluLayer, SoftmaxWithLossLayer, SigmoidLayer, LayerType
from core.loss import cross_entropy_error


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


class TwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        """
        x: input data
        t: answer label
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        x: input data
        t: answer label
        """

        loss_W = lambda _: self.loss(x, t)

        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = numerical_gradient(loss_W, self.params[key])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        # print(grads['W1'][0][:10]) -> 0 밖에 안나온다. 이유를 찾아야 한다.

        return grads


class BackpropagationTwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, use_xavier_init=False):
        # 가중치 표준편차 초기값
        if use_xavier_init:
            weight_init_std_w1 = 1 / np.sqrt(input_size)
            weight_init_std_w2 = 1 / np.sqrt(hidden_size)
        else:
            weight_init_std_w1 = weight_init_std
            weight_init_std_w2 = weight_init_std

        self.params = {
            'W1': weight_init_std_w1 * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std_w2 * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLossLayer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {
            'W1': self.layers['Affine1'].dW,
            'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW,
            'b2': self.layers['Affine2'].db
        }

        return grads

    def numerical_gradient(self, x, t):
        loss_W = lambda _: self.loss(x, t)

        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2'])
        }
        return grads


class ActivationType:
    Sigmoid = 'sigmoid'
    Relu = 'relu'


class MultiLayerNet:
    def __init__(self, input_size: int, hidden_size_list: list, output_size: int,
                 *, weight_init_std_deviation=0.01, use_xavier_init=False, use_he_init=False,
                 activation=ActivationType.Sigmoid, weight_decay_lambda=0):

        assert not (use_xavier_init and use_he_init), 'both use_xavier_init or use_he_init can not be True'
        all_size_list = [input_size] + hidden_size_list + [output_size]

        self._weight_decay_lambda = weight_decay_lambda
        self.params = []
        self._layers = []
        for i in range(len(all_size_list)-1):
            this_layer_size = all_size_list[i]
            next_layer_size = all_size_list[i + 1]

            # if use_xavier_init or use_he_init is True, weight_init_std value is override
            if use_xavier_init:
                std_deviation = np.sqrt(1.0 / this_layer_size)
            elif use_he_init:
                std_deviation = np.sqrt(2.0 / this_layer_size)
            else:
                std_deviation = weight_init_std_deviation

            param = {
                'W': std_deviation * np.random.randn(this_layer_size, next_layer_size),
                'b': np.zeros(next_layer_size)
            }
            self.params.append(param)
            self._layers.append(AffineLayer(param['W'], param['b']))

            if i < (len(all_size_list) - 2):
                # 마지막 layer는 activation function을 적용하지 않는다.
                if activation == ActivationType.Sigmoid:
                    self._layers.append(SigmoidLayer())
                elif activation == ActivationType.Relu:
                    self._layers.append(ReluLayer())
                else:
                    assert False, f'{activation} activation is not a proper value'

        self._last_layer = SoftmaxWithLossLayer()

    def predict(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)

        weight_decay = 0
        if self._weight_decay_lambda:
            for param in self.params:
                W = param['W']
                weight_decay += 0.5 * self._weight_decay_lambda * np.sum(W ** 2)
        return self._last_layer.forward(y, t) + weight_decay

    def gradient(self, x, t):

        # forward propagation
        self.loss(x, t)

        # back propagation
        # init value is 1
        dout = self._last_layer.backward(1)
        for layer in reversed(self._layers):
            dout = layer.backward(dout)

        gradients = []
        for layer in self._layers:
            if layer.type == LayerType.Affine:
                gradient = {
                    'W': layer.dW + self._weight_decay_lambda * layer.W,
                    'b': layer.db
                }
                gradients.append(gradient)

        return gradients

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])
