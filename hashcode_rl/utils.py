import torch
from torch import nn as nn, distributions as distributions
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.reshape(-1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def train(env, policy, optimizer, discount_factor, device):
    policy.train()

    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        action_pred = policy(state)

        action_prob = F.softmax(action_pred, dim=-1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)

        state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat([shit.unsqueeze(0) for shit in log_prob_actions], dim=0)

    returns = calculate_returns(rewards, discount_factor, device)

    loss = update_policy(returns, log_prob_actions, optimizer)

    return loss, episode_reward


def calculate_returns(rewards, discount_factor, device, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns).to(device)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def update_policy(returns, log_prob_actions, optimizer):
    returns = returns.detach()

    loss = -(returns * log_prob_actions).sum()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


def evaluate(env, policy, device):
    policy.eval()

    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_pred = policy(state)

            action_prob = F.softmax(action_pred, dim=-1)

        action = torch.argmax(action_prob, dim=-1)

        state, reward, done, _ = env.step(action.item())

        episode_reward += reward

    return episode_reward
