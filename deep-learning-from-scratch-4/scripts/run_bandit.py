from src.bandit.bandit import Bandit

if __name__ == "__main__":
    num = 3
    bandit = Bandit(num)

    for i in range(num):
        for _ in range(10):
            print(f"slot machine {i+1}: {bandit.play(i)}")
