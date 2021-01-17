
class SGD:
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate

    def update(self, params: dict, grads: dict):
        for key in params.keys():
            params[key] -= self._learning_rate * grads[key]
