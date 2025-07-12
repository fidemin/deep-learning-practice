import numpy as np
from typeguard import typechecked


class Bandit:
    @typechecked
    def __init__(
        self,
        num_of_machines: int,
    ):
        self._machine_rates = np.random.rand(num_of_machines)

    def play(self, machine: int):
        rate = self._machine_rates[machine]

        if rate > np.random.rand():
            return 1
        else:
            return 0
