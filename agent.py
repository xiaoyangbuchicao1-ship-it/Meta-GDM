import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import ContextEncoder, SocialAttention


class Actor(nn.Module):

    def __init__(self, z_dim, action_dim=1, hidden_dim=256, num_heads=4):
        super().__init__()
        self.agent_feat_dim = 1 + 1 + 1 + z_dim
        self.social_attn = SocialAttention(self.agent_feat_dim, hidden_dim, num_heads=num_heads)
        decision_input_dim = hidden_dim + self.agent_feat_dim
        self.l1 = nn.Linear(decision_input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, z):
        batch_size = state.shape[0]
        num_agents = z.shape[1]
        state_reshaped = state.view(batch_size, num_agents, 3)
        current_opinions = state_reshaped[:, :, 0].unsqueeze(2)
        distance_to_mean = state_reshaped[:, :, 1].unsqueeze(2)
        time_left = state_reshaped[:, :, 2].unsqueeze(2)
        agent_inputs = torch.cat([current_opinions, distance_to_mean, time_left, z], dim=2)

        group_context = self.social_attn(agent_inputs)
        group_context = group_context.unsqueeze(1).expand(-1, num_agents, -1)
        actor_input = torch.cat([group_context, agent_inputs], dim=-1)

        x = F.relu(self.l1(actor_input))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state, z):
        mean, log_std = self.forward(state, z)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1).sum(dim=1, keepdim=True)
        return y_t.squeeze(-1), log_prob, torch.tanh(mean).squeeze(-1)


class Critic(nn.Module):

    def __init__(self, z_dim, action_dim=1, hidden_dim=256, num_heads=4):
        super().__init__()
        self.agent_feat_dim = 1 + 1 + 1 + z_dim
        self.critic_feat_dim = self.agent_feat_dim + action_dim
        self.attn1 = SocialAttention(self.critic_feat_dim, hidden_dim, num_heads=num_heads)
        self.attn2 = SocialAttention(self.critic_feat_dim, hidden_dim, num_heads=num_heads)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, z, action):
        batch_size = state.shape[0]
        num_agents = z.shape[1]
        state_reshaped = state.view(batch_size, num_agents, 3)
        current_opinions = state_reshaped[:, :, 0].unsqueeze(2)
        distance_to_mean = state_reshaped[:, :, 1].unsqueeze(2)
        time_left = state_reshaped[:, :, 2].unsqueeze(2)
        agent_inputs = torch.cat([current_opinions, distance_to_mean, time_left, z], dim=2)
        if action.dim() == 2:
            action = action.unsqueeze(2)

        critic_inputs = torch.cat([agent_inputs, action], dim=-1)
        q1_context = self.attn1(critic_inputs)
        q2_context = self.attn2(critic_inputs)
        q1 = self.l2(F.relu(self.l1(q1_context)))
        q2 = self.l4(F.relu(self.l3(q2_context)))
        return q1, q2


class MetaAgent:

    def __init__(self, z_dim=8, hidden_dim=256, lr=3e-4,
                 gamma=0.95, tau=0.01, alpha=0.1,
                 context_window=10, kl_weight=0.05,
                 num_heads=4, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.z_dim = z_dim
        self.kl_weight = kl_weight
        self.context_window = context_window
        self.num_heads = num_heads

        self.encoder = ContextEncoder(input_dim=3, z_dim=z_dim, num_heads=num_heads).to(device)
        self.actor = Actor(z_dim=z_dim, hidden_dim=hidden_dim, num_heads=num_heads).to(device)
        self.critic = Critic(z_dim=z_dim, hidden_dim=hidden_dim, num_heads=num_heads).to(device)
        self.critic_target = Critic(z_dim=z_dim, hidden_dim=hidden_dim, num_heads=num_heads).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, context, evaluate=False, exploration_noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        context = torch.FloatTensor(context).to(self.device)

        with torch.no_grad():
            context_batch = context.unsqueeze(0)
            z, _, _, _ = self._encode_context(context_batch, sample=False)
            if evaluate:
                _, _, action = self.actor.sample(state, z)
            else:
                action, _, _ = self.actor.sample(state, z)

        action = action.cpu().numpy()[0]
        if exploration_noise > 0:
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def set_eval_mode(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def set_train_mode(self):
        self.encoder.train()
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def update(self, batch):
        context, state, action, reward, next_state, next_context, not_done = batch

        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        elif reward.dim() == 2:
            reward = reward.mean(dim=1, keepdim=True)
        else:
            reward = reward.view(reward.size(0), -1).mean(dim=1, keepdim=True)
        not_done = not_done.view(-1, 1)

        z, mu, std, kl_loss = self._encode_context(context, sample=True)

        with torch.no_grad():
            next_z, _, _, _ = self._encode_context(next_context, sample=False)
            next_action, next_log_pi, _ = self.actor.sample(next_state, next_z)
            q1_next, q2_next = self.critic_target(next_state, next_z, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            next_q_value = reward + not_done * self.gamma * min_q_next

        q1, q2 = self.critic(state, z, action)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        encoder_total_loss = critic_loss + self.kl_weight * kl_loss

        self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        encoder_total_loss.backward()
        self.encoder_optimizer.step()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state, z.detach())
        q1_pi, q2_pi = self.critic(state, z.detach(), pi)
        policy_loss = ((self.alpha * log_pi) - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": policy_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def _encode_context(self, context, sample=True):
        batch_size, num_agents, seq_len, feat_dim = context.shape
        flat_context = context.view(-1, seq_len, feat_dim)
        mu, std = self.encoder(flat_context)
        if sample:
            z_flat = self.encoder.sample_z(mu, std)
            kl_loss = -0.5 * torch.sum(
                1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2),
                dim=1
            ).mean()
        else:
            z_flat = mu
            kl_loss = torch.zeros((), device=context.device)
        z = z_flat.view(batch_size, num_agents, -1)
        return z, mu, std, kl_loss

    def save_checkpoint(self, filepath, episode, extra=None):
        extra = extra or {}
        checkpoint = {
            "episode": episode,
            "encoder_state_dict": self.encoder.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "z_dim": self.z_dim,
            "context_window": self.context_window,
            "num_heads": self.num_heads,
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            **extra
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "critic_target_state_dict" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())
        return checkpoint
