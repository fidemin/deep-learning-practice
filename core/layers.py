import enum

import numpy as np

from core.activation import softmax
from core.loss import cross_entropy_error


class LayerType:
    Affine = 'affine'
    Activation = 'activation'
    Dropout = 'dropout'
    Loss = 'loss'
    Other = 'others'


class MulLayer:
    type = LayerType.Other

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    type = LayerType.Other

    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class ReluLayer:
    type = LayerType.Activation

    def __init__(self):
        self.x = None
        self._mask = None

    def forward(self, x):
        self._mask = (x <= 0)
        out = x.copy()
        out[self._mask] = 0
        return out

    def backward(self, dout):
        dout[self._mask] = 0
        dx = dout
        return dx


class SigmoidLayer:
    type = LayerType.Activation

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx


class AffineLayer:
    type = LayerType.Affine

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # for tensor
        self.original_x_shape = x.shape
        # x의 차원이 2차원 배열 이상이면, (N, ?) 형태의 배열로 reshape 한다.
        # 즉, 배치의 각 요소가 1차원 배열이 아니면, 1차원 형태로 flatten 한다고 보면 된다.
        x = x.reshape(x.shape[0], -1)

        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        # self.x.T의 dimension은 self.original_x_shape와 동일하다는 것을 인지하자.
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


class DropoutLayer:
    type = LayerType.Dropout

    def __init__(self, dropout_ratio=0):
        self._dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True):
        if self._dropout_ratio and train_flag:
            self.mask = np.random.rand(*x.shape) > self._dropout_ratio
            return x * self.mask
        else:
            # 훈련이 아닌 경우 그대로 내보낸다.
            return x

    def backward(self, dout):
        return dout * self.mask


class SoftmaxWithLossLayer:
    type = LayerType.Loss

    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 데이터 당 1개의 오차를 전달한다.
        # Why? 수식적으로 batch_size로 나누어 주는게 맞다. (numerical_gradient 값과 오차가 거의 안난다.)
        # https://stackoverflow.com/questions/65275522/why-is-softmax-classifier-gradient-divided-by-batch-size-cs231n
        # batch_size로 나누지 않았을 경우, gradient 값이 너무 커지기 때문에, 수렴하지 않는다.
        # batch_size / 2 로 나누었을 경우 (gradient이 2배 가량 큰 경우), 약간 overfitting 된다.
        # batch_size * 2 로 나누었을 경우, 약간 underfitting 된다.
        dx = (self.y - self.t) / batch_size
        return dx * dout
