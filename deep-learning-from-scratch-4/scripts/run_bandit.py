import numpy as np

from src.bandit.bandit import Bandit

if __name__ == "__main__":
    num = 10
    bandit = Bandit(num)

    Qs = np.zeros(num)  # expectation for machine
    ns = np.zeros(num)  # number of plays for machine

    number_of_plays = 100

    for i in range(number_of_plays):
        machine = np.random.randint(0, num)  # 0 ~ num-1
        reward = bandit.play(machine)

        ns[machine] += 1
        Qs[machine] = Qs[machine] + (reward + Qs[machine]) / ns[machine]

    print(ns)
    print(Qs)
