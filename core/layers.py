import numpy as np

from core.activation import softmax
from core.loss import cross_entropy_error


class MulLayer:
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
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class ReluLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out (1.0 - self.out)
        return dx


class AffineLayer:
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


class SoftmaxWithLossLayer:
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
        # Why? (y - t)의 식은 batch size가 1이라고 가정하고 계산했을 때 결과이다.
        # 역전파 시, dout=1부터 시작하므로 batch_size로 나누어야 한다.
        dx = (self.y - self.t) / batch_size
        return dx * dout
