import matplotlib.pyplot as plt
import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device(device="mps")
elif torch.cuda.is_available():
    DEVICE = torch.device(device="cuda")
else:
    DEVICE = torch.device(device="cpu")

def plot_rewards(rewards_list):
    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Across Episodes')
    plt.show()