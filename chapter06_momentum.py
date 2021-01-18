import matplotlib.pyplot as plt
import numpy as np

# gradient descent로 x**2 의 최소값을 구하는 과정에서, momentum을 적용해본다.
# 적용하지 않았을 때 보다 더 빠르게 수렴한다.

def gradient(x):
    return 2 * x


def momentum(dx, v):
    return 0.8 * v - 0.01 * dx


def new_position(x, v):
    return x + v


x = 3
v = 0
count = 0
x_list = []
while True:
    x_list.append(x)
    print('x: ', x)
    dx = gradient(x)
    if abs(dx) < 0.0001:
        break
    v = momentum(dx, v)
    x = new_position(x, v)
    print('v: ', v)
    print('-----------')
    count += 1

print('total iteration: ', count)
plt.plot(np.arange(len(x_list)), x_list, label='x values by iteration: x ** 2')
plt.xlabel("# of iters")
plt.ylabel("x value")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()



