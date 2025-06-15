from ppo import Agent, PygameVisualizer
import numpy as np
import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pygame

render = True
visualizer = PygameVisualizer() if render else None

env = gym.make("MountainCarContinuous-v0")
input_dims = env.observation_space.shape
n_actions = env.action_space.shape[0]

agent = Agent(
    input_dims, 
    n_actions, 
    lr=3e-4,
    gamma=0.995,
    gae_lambda=0.95,
    policy_clip=0.1,
    batch_size=256,
    n_epochs=15,
    entropy_coef=0.02
)

N = 10
n_games = 100
score_history = []
best_score = -np.inf
success_count = 0
frame_count = 0

agent.load_models()

for i in range(n_games):
    obs, _ = env.reset()
    done = truncated = False
    score = 0
    step_count = 0
    agent.last_position = -1.2 

    while not (done or truncated):
        action, prob, val = agent.next_action(obs)
        action = np.clip(action, -1, 1)
        
        next_obs, _, done, truncated, _ = env.step(action)

        position, velocity = next_obs
        step_count += 1
        frame_count += 1
        
        progress = position - agent.last_position
        reward = 7 * progress
        
        reward += 4 * (position + 1)
        
        if (position < 0 and velocity > 0):
            reward += 5 * abs(velocity) 
        
        if (position >= 0 and velocity < 0):
            reward += 5 * abs(velocity) 
        
        if position >= 0.45:
            reward += 1500
            success_count += 1
            
        if progress < -0.01:
            reward -= 1.0
            
        if abs(progress) < 0.001 and step_count > 50:
            reward -= 0.5
            
        if abs(velocity) < 0.001 and step_count > 100:
            reward -= 1.0
            
        agent.last_position = position

        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if visualizer:
                        visualizer.close()
                    env.close()
                    exit()
            visualizer.render(position, velocity, frame_count, lambda x : np.sin(2*x)) #tutaj wstawiamy funkcje na ktorej trenujemy nasz model

        reward-=step_count*(0.0055)
        agent.remember(obs, action, prob, val, reward, done or truncated)

        obs = next_obs
        score += reward

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if i % N == 0: 
        agent.learn()

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()


    print(f'Epizod {i}, Wynik: {score:.2f}, Śr. wynik: {avg_score:.2f}, Sukcesy: {success_count}')

env.close()
if visualizer:
    visualizer.close()
