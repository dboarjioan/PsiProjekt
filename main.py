from klasy import Agent, PygameVisualizer
import numpy as np
import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pygame
import math

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def run_demo(agent, num_episodes=20):
    env = gym.make("MountainCarContinuous-v0")
    visualizer = PygameVisualizer()
    frame_count = 0
    success_count = 0
    
    for i in range(num_episodes):
        render = True
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
            
            if position >= 0.45:
                success_count += 1
                print(f"SUKCES {success_count}")
                
            agent.last_position = position

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if visualizer:
                            visualizer.close()
                        env.close()
                        exit()
                visualizer.render(position, velocity, frame_count)

            obs = next_obs


    print(f'Epizod {i}, Sukcesy: {success_count}')

    print("\n" + "="*50)
    print(f"Podsumowanie demonstracji:")
    print(f"Liczba epizodów: {num_episodes}")
    print(f"Liczba sukcesów: {success_count}")
    print(f"Wskaźnik sukcesu: {success_count/num_episodes*100:.1f}%")
    print("="*50)
    
    visualizer.close()
    env.close()

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    env.close()

    agent = Agent(
    input_dims, 
    n_actions, 
    lr=3e-4,
    gamma=0.995,
    gae_lambda=0.95,
    policy_clip=0.1,
    batch_size=16,
    n_epochs=15,
    entropy_coef=0.02
    )

    agent.load_models()
    
    model_path = "modelsPath/actor.pth"
    
    run_demo(agent, num_episodes=20)
