import numpy as np
import torch
from torch.distributions import Categorical
from model import *
from replay import *


class Agent:
    def __init__(self, cfg) -> None:
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.actor = Actor(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs  # update policy for K epochs, or sampling times
        self.eps_clip = cfg.eps_clip  # clip parameter for PPO epsilon
        self.entropy_coef = cfg.entropy_coef  # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq

    def sample_action(self, state):
        self.sample_count += 1  # one more env step
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)  # 在第0层增加一个维度
        probs = self.actor(state)
        dist = Categorical(probs)  # the distribution of probs, 离散分布
        action = dist.sample()  # sample from the distribution
        self.log_probs = dist.log_prob(action).detach()  # the log probability of the action
        return action.detach().cpu().numpy().item()

    @torch.no_grad()
    # 其中的数据不需要计算梯度，也不会进行反向传播
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        action = torch.argmax(probs)
        # dist = Categorical(probs)
        # action = dist.sample()
        return action.detach().cpu().numpy().item()

    @staticmethod
    def compute_gae(next_value, rewards, dones, values, gamma=0.99, tau=0.95):
        """compute the actual discounted rewards through GAE"""
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1-dones[step]) - values[step]
            gae = delta + gamma * tau * (1-dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, final_state):
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()  # 只采样一次，采样全部的transition
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(old_log_probs), device=self.device, dtype=torch.float32)

        # compute the advantages
        old_values = self.critic(old_states)
        old_values = old_values.squeeze(dim=1).detach().cpu().numpy().tolist()
        final_state = torch.FloatTensor(final_state).to(self.device)
        final_value = self.critic(final_state)
        returns = self.compute_gae(final_value.detach().cpu().numpy().item(), old_rewards, old_dones, old_values)

        returns = torch.tensor(np.array(returns), device=self.device, dtype=torch.float32)
        values = torch.tensor(np.array(old_values), device=self.device, dtype=torch.float32)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # normalizing, 1e-5 to avoid division by zero

        # update policy for K epochs
        for _ in range(self.k_epochs):
            values = self.critic(old_states)
            probs = self.actor(old_states)
            dist = Categorical(probs)
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta_old):
            ratio = torch.exp(new_probs - old_log_probs)  # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

    def save_model_state(self, output_dir):
        torch.save(self.actor.state_dict(), output_dir+'_actor.ckpt')
        torch.save(self.critic.state_dict(), output_dir + '_critic.ckpt')

    def load_model_state(self, file: tuple):
        self.actor.load_state_dict(torch.load(file[0]))
        self.actor.eval()
        self.critic.load_state_dict(torch.load(file[1]))
        self.critic.eval()
