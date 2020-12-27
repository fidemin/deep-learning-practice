import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        # if train data is one-hot vector (t total size == y total size)
        # change t to index of answer layer 
        # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] -> 2
        # why? return 에서 사용하는 함수를 공통으로 사용하기 위해서
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

