import numpy as np
from typeguard import typechecked


class Bandit:
    @typechecked
    def __init__(
        self,
        num_of_machines: int,
        *,
        use_noise: bool = False,
    ):
        self._num_of_machines = num_of_machines
        self._machine_rates = np.random.rand(num_of_machines)
        self._use_noise = use_noise

    def play(self, machine: int):
        rate = self._machine_rates[machine]

        # add noise
        if self._use_noise:
            self._machine_rates += 0.1 * np.random.randn(self._num_of_machines)

        if rate > np.random.rand():
            return 1
        else:
            return 0
