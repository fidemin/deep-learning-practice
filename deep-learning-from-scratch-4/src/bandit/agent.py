import numpy as np
from typeguard import typechecked


class Agent:
    @typechecked
    def __init__(
        self,
        action_size: int,
        *,
        epsilon: float = 0.1,
        alpha: float | None = None
    ):
        self._action_size = action_size
        self._epsilon = epsilon
        self._Qs = np.zeros(action_size)
        self._ns = np.zeros(action_size)
        self._alpha = alpha

    def update(self, action: int, reward: int | float):
        self._ns[action] += 1

        alpha = self._alpha if self._alpha else self._ns[action]
        self._Qs[action] = (
            self._Qs[action] + (reward - self._Qs[action]) / alpha
        )

    def get_action(self):
        if np.random.rand() < self._epsilon:
            return np.random.randint(0, self._action_size)

        return np.argmax(self._Qs)
