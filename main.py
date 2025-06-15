import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import pygame
import torch as T
from klasy import Agent, PygameVisualizer

class CustomMountainCarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, custom_func=None):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.terrain_func = custom_func if custom_func is not None else lambda x: math.sin(3 * x) * 0.45
        self.state = None

    def _height(self, x):
        return self.terrain_func(x)

    def _get_slope(self, x, eps=1e-5):
        return (self.terrain_func(x + eps) - self.terrain_func(x - eps)) / (2 * eps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = self.state
        force = np.clip(action[0], -1.0, 1.0)

        slope = self._get_slope(position)  
        velocity += force * self.power
        velocity += -0.0025 * slope  

        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position)
        reward = 100.0 if done else -0.1 * abs(force)

        self.state = np.array([position, velocity])
        return self.state.copy(), reward, done, False, {}

    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


register(
    id='CustomMountainCarContinuous-v0',
    entry_point='__main__:CustomMountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)


def run_demo(agent, func, num_episodes=20):
    env = gym.make("CustomMountainCarContinuous-v0", custom_func=func)
    visualizer = PygameVisualizer()
    frame_count = 0
    success_count = 0

    for i in range(num_episodes):
        render = True
        obs, _ = env.reset()
        done = truncated = False
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
                visualizer.render(position, velocity, frame_count, func)

            obs = next_obs

        print(f"Epizod {i}, Sukcesy: {success_count}")

    print("\n" + "="*50)
    print(f"Podsumowanie demonstracji:")
    print(f"Liczba epizodów: {num_episodes}")
    print(f"Liczba sukcesów: {success_count}")
    print(f"Wskaźnik sukcesu: {success_count/num_episodes*100:.1f}%")
    print("="*50)

    visualizer.close()
    env.close()


if __name__ == "__main__":
    env = gym.make("CustomMountainCarContinuous-v0", custom_func=lambda x: np.sin(3 * x) * 0.45)
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

    def smooth_transition(x, x0, k=10):
        return 1 / (1 + np.exp(-k*(x-x0)))

    functions = [
        lambda x: np.abs(1.4*(x+0.1))-0.7,
        lambda x: 0.5*np.sin(6 * x) if x > -0.5 else -2*x - 1.05,
        lambda x: 0.5 * np.exp(np.sin(6 * (x + 0.45))) * np.cos(6 * (x + 0.45)) if x > -0.7  else x*x -0.48,
        lambda x: 0.3 * np.tanh(3 * x) - 0.2 * np.exp(-6 * (x + 0.5) ** 2),
        lambda x: (
        (0.3 * np.sin(3*x) * np.exp(-0.2*x**2)) * smooth_transition(x, -0.2) +
        (0.5 * np.tanh(5*(x+0.4))) * (1 - smooth_transition(x, -0.2)) * smooth_transition(x, -0.6) +
        (-0.8*x - 0.3) * (1 - smooth_transition(x, -0.6))
        )
    ] 

    for func in functions:
        run_demo(agent, func, num_episodes=10)
