import numpy as np


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    with np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) as it:
        for _ in it:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            x[idx] = tmp_val # 값 복원  
    return grad
