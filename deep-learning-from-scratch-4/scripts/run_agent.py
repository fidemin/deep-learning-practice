import matplotlib.pyplot as plt
import numpy as np

from src.bandit.agent import Agent
from src.bandit.bandit import Bandit

if __name__ == "__main__":
    np.random.seed(0)
    steps = 1000
    epsilon = 0.1

    num_of_actions = 10
    bandit = Bandit(num_of_actions)
    agent = Agent(num_of_actions, epsilon=epsilon)

    agent_total_reward = 0
    random_total_reward = 0

    agent_total_rewards = []
    random_total_rewards = []

    agent_rates = []
    random_rates = []

    for step in range(steps):
        agent_action = agent.get_action()
        agent_reward = bandit.play(agent_action)

        random_action = np.random.randint(0, num_of_actions)
        random_reward = bandit.play(random_action)

        agent.update(agent_action, agent_reward)

        agent_total_reward += agent_reward
        agent_total_rewards.append(agent_total_reward)
        agent_rates.append(agent_total_reward / (step + 1))

        random_total_reward += random_reward
        random_total_rewards.append(random_total_reward)
        random_rates.append(random_total_reward / (step + 1))

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(agent_total_rewards, label="Agent Total Reward")
    ax1.plot(random_total_rewards, label="Random Total Reward")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Total Reward Comparison")
    ax1.legend()

    ax2.plot(agent_rates, label="Agent Rate")
    ax2.plot(random_rates, label="Random Rate")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Rates")
    ax2.set_title("Reward Rates Comparison")
    ax2.legend()

    plt.tight_layout()
    plt.show()
