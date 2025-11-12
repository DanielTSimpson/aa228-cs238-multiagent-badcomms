from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

class FireSearchEnv(Env):
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.observation_space = MultiDiscrete([self.grid_size]*4)  # (x1, y1, x2, y2)
        self.action_space = MultiDiscrete([5, 5])  # both drones: up, down, left, right, stay
        self.fire_pos = np.random.randint(0, self.grid_size, size=2)
        self.drones = [(0, 0), (self.grid_size-1, self.grid_size-1)]
        self.fig, self.ax = None, None
        self.reset()

    def reset(self):
        self.drones = [(0, 0), (self.grid_size-1, self.grid_size-1)]
        self.fire_pos = np.random.randint(0, self.grid_size, size=2)
        return np.array([*self.drones[0], *self.drones[1]])

    def move(self, pos, action):
        # 0=up, 1=down, 2=left, 3=right, 4=stay
        x, y = pos
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.grid_size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.grid_size - 1, y + 1)
        return (x, y)

    def step(self, actions):
        # Update both drones
        self.drones = [self.move(self.drones[i], actions[i]) for i in range(2)]

        done = any(drone == tuple(self.fire_pos) for drone in self.drones)
        reward = 1.0 if done else -0.01
        obs = np.array([*self.drones[0], *self.drones[1]])
        return obs, reward, done, {}, {}

    def render(self):
        # Build grid representation
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[tuple(self.fire_pos)] = 3  # fire
        grid[self.drones[0]] = 1  # drone 1
        grid[self.drones[1]] = 2  # drone 2

        cmap = colors.ListedColormap(['white', 'blue', 'red', 'orange'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(grid, cmap=cmap, norm=norm)
            self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.ion()
            plt.show(block=False)
        else:
            self.im.set_data(grid)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        time.sleep(1)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None


if __name__ == '__main__':
    env = FireSearchEnv(grid_size=5)
    obs = env.reset()

    for step in range(30):
        actions = env.action_space.sample()  # random move for both drones
        obs, reward, done, _, _ = env.step(actions)
        env.render()

        if done:
            print(f"🔥 Fire found at step {step}! Reward: {reward}")
            break

    env.close()
