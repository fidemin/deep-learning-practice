import matplotlib.pyplot as plt
import numpy as np

from src.bandit.agent import Agent
from src.bandit.bandit import Bandit

if __name__ == "__main__":
    runs = 200
    steps = 1000

    num_of_actions = 10

    epsilons = [1.0, 0.5, 0.3, 0.1]
    all_rates = np.zeros((len(epsilons), runs, steps))

    for eps_i, epsilon in enumerate(epsilons):
        for run in range(runs):
            bandit = Bandit(num_of_actions)
            agent = Agent(num_of_actions, epsilon=epsilon)

            agent_total_reward = 0

            agent_total_rewards = []

            agent_rates = []
            random_rates = []

            for step in range(steps):
                agent_action = agent.get_action()
                agent_reward = bandit.play(agent_action)
                agent.update(agent_action, agent_reward)

                agent_total_reward += agent_reward
                agent_total_rewards.append(agent_total_reward)
                agent_rates.append(agent_total_reward / (step + 1))

            all_rates[eps_i][run] = agent_rates

    avg_agent_rates = np.average(all_rates, axis=1)

    for eps_i, epsilon in enumerate(epsilons):
        plt.plot(avg_agent_rates[eps_i], label=f"epsilon: {epsilon}")

    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.title("Reward Rates Comparison")
    plt.legend()

    plt.show()
