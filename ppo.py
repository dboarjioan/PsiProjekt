import numpy as np
import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pygame
import math
import os
import time

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=512, fc2_dims=512):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Parameter(T.zeros(n_actions))

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mu.weight, gain=0.01)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=512, fc2_dims=512):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.v(x)
        return value

class Agent:
    def __init__(self, input_dims, n_actions, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.1, batch_size=16, n_epochs=15, entropy_coef=0.02):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        self.actor = ActorNetwork(input_dims, n_actions).to(device)
        self.critic = CriticNetwork(input_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        self.memory = []
        self.last_position = -1.2

    def remember(self, state, action, prob, val, reward, done):
        self.memory.append((state, action, prob, val, reward, done))

    def clear_memory(self):
        self.memory = []

    def next_action(self, state):
        state = T.tensor(np.array([state]), dtype=T.float32).to(device)
        mu, std = self.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(state)
        return action.cpu().numpy()[0], log_prob.item(), value.item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, old_probs, values, rewards, dones = zip(*self.memory)

        states = T.tensor(np.array(states), dtype=T.float32).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float32).to(device)
        old_probs = T.tensor(old_probs, dtype=T.float32).unsqueeze(1).to(device)
        values = T.tensor(values, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        dones = T.tensor(dones, dtype=T.float32).to(device)

        returns = T.zeros_like(rewards).to(device)
        advantages = T.zeros_like(rewards).to(device)
        running_return = 0
        running_advantage = 0
        previous_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * previous_value * mask - values[t]
            running_advantage = delta + self.gamma * self.gae_lambda * mask * running_advantage
            advantages[t] = running_advantage
            running_return = rewards[t] + self.gamma * running_return * mask
            returns[t] = running_return
            previous_value = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for i in range(0, len(states), self.batch_size):
                end = i + self.batch_size
                batch_indices = indices[i:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                mu, std = self.actor(batch_states)
                dist = Normal(mu, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                ratio = (new_log_probs - batch_old_probs).exp()

                weighted_probs = ratio * batch_advantages.unsqueeze(1)
                clipped_probs = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages.unsqueeze(1)
                
                entropy = dist.entropy().mean()
                actor_loss = -T.min(weighted_probs, clipped_probs).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                critic_value = self.critic(batch_states).squeeze()
                critic_loss = F.smooth_l1_loss(critic_value, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        self.clear_memory()

    def save_models(self, directory="modelsPath"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        T.save(self.actor.state_dict(), os.path.join(directory, 'actor.pth'))
        T.save(self.critic.state_dict(), os.path.join(directory, 'critic.pth'))
        print("Zapisano modele")

    def load_models(self, directory="modelsPath"):
        self.actor.load_state_dict(T.load(os.path.join(directory, 'actor.pth')))
        self.critic.load_state_dict(T.load(os.path.join(directory, 'critic.pth')))
        print("Wczytano modele")

class PygameVisualizer:
    def __init__(self, width=800, height=500):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MountainCarContinuous PPO")
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        
        self.car_width = 40
        self.car_height = 20
        self.wheel_radius = 8
        
        self.car_surface = pygame.Surface((self.car_width, self.car_height + self.wheel_radius), pygame.SRCALPHA)
        pygame.draw.rect(self.car_surface, (220, 50, 50), (0, self.wheel_radius, self.car_width, self.car_height))
        pygame.draw.ellipse(self.car_surface, (70, 130, 180), 
                           (self.car_width//4, 0, self.car_width//2, self.car_height//2 + self.wheel_radius))
        pygame.draw.circle(self.car_surface, (40, 40, 40), (self.car_width//4, self.wheel_radius), self.wheel_radius)
        pygame.draw.circle(self.car_surface, (40, 40, 40), (3*self.car_width//4, self.wheel_radius), self.wheel_radius)
        pygame.draw.circle(self.car_surface, (100, 100, 100), 
                          (self.car_width//4, self.wheel_radius), self.wheel_radius//2)
        pygame.draw.circle(self.car_surface, (100, 100, 100), 
                          (3*self.car_width//4, self.wheel_radius), self.wheel_radius//2)
        
        self.background = pygame.Surface((width, height))
        self.background.fill((135, 206, 235))  
        
        for i in range(3):
            pygame.draw.polygon(self.background, (120 - i*20, 80 - i*20, 50 - i*10), [
                (0, height),
                (width, height),
                (width//2, height//3 - i*50)
            ])
        
        self.cloud_surface = pygame.Surface((width, 100), pygame.SRCALPHA)
        for i in range(5):
            x = (i * 200) % width
            pygame.draw.circle(self.cloud_surface, (255, 255, 255, 180), (x, 30), 20)
            pygame.draw.circle(self.cloud_surface, (255, 255, 255, 180), (x+15, 20), 25)
            pygame.draw.circle(self.cloud_surface, (255, 255, 255, 180), (x+30, 30), 20)

    def render(self, position, velocity, frame_count):
        self.screen.blit(self.background, (0, 0))
        
        cloud_offset = (frame_count // 2) % self.width
        self.screen.blit(self.cloud_surface, (-cloud_offset, 50))
        self.screen.blit(self.cloud_surface, (self.width - cloud_offset, 50))
        
        pygame.draw.rect(self.screen, (34, 139, 34), (0, self.height - 50, self.width, 50))
        
        points = []
        for x in range(self.width):
            pos = -1.2 + (x / self.width) * (0.6 + 1.2)
            y = np.sin(3 * pos) * 0.45 + 0.55
            screen_y = self.height - 50 - int(y * (self.height - 150) / 1.0)
            points.append((x, screen_y))
        pygame.draw.lines(self.screen, (101, 67, 33), False, points, 12)
        
        goal_x = int((0.45 + 1.2) / 1.8 * self.width)
        goal_y = self.height - 50 - int((np.sin(3 * 0.45) * 0.45 + 0.55) * (self.height - 150) / 1.0)
        pygame.draw.line(self.screen, (0, 0, 0), (goal_x, goal_y - 30), (goal_x, goal_y - 10), 3)
        pygame.draw.polygon(self.screen, (255, 0, 0), [
            (goal_x, goal_y - 30),
            (goal_x, goal_y - 20),
            (goal_x + 25, goal_y - 25)
        ])
        
        car_x = int((position + 1.2) / 1.8 * self.width)
        track_y = np.sin(3 * position) * 0.45 + 0.55
        car_y = self.height - 50 - int(track_y * (self.height - 150) / 1.0)
        
        slope = 1.35 * math.cos(3 * position)
        scale_x = self.width / 1.8
        scale_y = (self.height - 150) / 1.0
        screen_slope = -slope * (scale_y / scale_x)
        angle_radians = math.atan(screen_slope)
        angle_degrees = math.degrees(angle_radians)
        
        rotated_car = pygame.transform.rotate(self.car_surface, angle_degrees)
        car_rect = rotated_car.get_rect(center=(car_x, car_y))
        
        shadow_alpha = 100 - abs(int(position * 30))
        shadow_alpha = max(30, min(100, shadow_alpha))
        shadow = pygame.Surface((car_rect.width, car_rect.height), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, shadow_alpha))
        self.screen.blit(shadow, (car_rect.x + 5, car_rect.y + 8))
        
        self.screen.blit(rotated_car, car_rect)
        
        font = pygame.font.SysFont('Arial', 24, bold=True)
        speed_text = f"Prędkość: {velocity:.2f}"
        pos_text = f"Pozycja: {position:.2f}"
        
        speed_surface = font.render(speed_text, True, (0, 0, 0))
        pos_surface = font.render(pos_text, True, (0, 0, 0))
        
        self.screen.blit(speed_surface, (10, 10))
        self.screen.blit(pos_surface, (10, 45))
        
        goal_surface = font.render("CEL", True, (255, 0, 0))
        self.screen.blit(goal_surface, (goal_x - 15, goal_y - 60))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
