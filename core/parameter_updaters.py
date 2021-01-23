import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate

    def update(self, params: dict, grads: dict):
        if type(params) == dict:
            for key in params.keys():
                params[key] -= self._learning_rate * grads[key]
        else:
            # params is array
            for i in range(len(params)):
                for key in params[i].keys():
                    params[i][key] -= self._learning_rate * grads[i][key]


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._v = None

    def update(self, params, grads):
        if type(params) == dict:
            if self._v is None:
                self._v = {}
                for key, val in params.items():
                    self._v[key] = np.zeros_like(val)

            for key in params.keys():
                self._v[key] = self._momentum * self._v[key] - self._learning_rate * grads[key]
                params[key] += self._v[key]
        else:
            if self._v is None:
                self._v = []
                for param in params:
                    v = {key: np.zeros_like(val) for key, val in param.items()}
                    self._v.append(v)

            for i in range(len(params)):
                for key in params[i].keys():
                    self._v[i][key] = self._momentum * self._v[i][key] - self._learning_rate * grads[i][key]
                    params[i][key] += self._v[i][key]


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate
        self._h = None

    def update(self, params: dict, grads: dict):
        if type(params) == dict:
            if self._h is None:
                self._h = {}
                for key, val in params.items():
                    self._h[key] = np.zeros_like(val)

            for key in params.keys():
                self._h[key] += grads[key] * grads[key]
                params[key] -= self._learning_rate * grads[key] / (np.sqrt(self._h[key]) + 1e-7)
        else:
            # params is array
            if self._h is None:
                self._h = []
                for param in params:
                    h = {
                        'W': np.zeros_like(param['W']),
                        'b': np.zeros_like(param['b'])
                    }
                    self._h.append(h)

            for i in range(len(params)):
                h = self._h[i]
                for key in h.keys():
                    h[key] += grads[i][key] * grads[i][key]
                    params[i][key] -= self._learning_rate * grads[i][key] / (np.sqrt(h[key] + 1e-7))