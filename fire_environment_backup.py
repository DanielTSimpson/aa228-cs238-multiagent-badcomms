from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class FireSearchEnv(Env):
    """
    Example environment on a grid space where F is the fire and D is the drone
    ^
    |   F
    |
    y       D
    |_x_______>
    """
    def __init__(self, grid_size = 10):
        self.grid_size = grid_size
        self.action_space = Discrete(5)
        self.fig, self.ax = None, None
        self.reset()

    def reset(self):
        self.fire_pos = np.random.randint(0, self.grid_size, size = 2)
        self.drone1 = np.random.randint(0, self.grid_size, size = 2)
        self.drone2 = np.random.randint(0, self.grid_size, size = 2)

        #If the fire and drone1 are on the same spot, reposition the drone
        while np.array_equal(self.fire_pos, self.drone1):
            self.drone1 = np.random.randint(0, self.grid_size, size = 2)

        #If the fire and drone1 are on the same spot, reposition the drone
        while np.array_equal(self.fire_pos, self.drone2) or np.array_equal(self.drone1, self.drone2):
            self.drone2 = np.random.randint(0, self.grid_size, size = 2)
    
    def move(self, state, action):
        # Actions include: 0:stay, 1:up, 2:down, 3:left, 4:right
        x, y = state
        if action == 1: y = min(self.grid_size - 1, y + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: x = max(0, x - 1)
        elif action == 4: x = min(self.grid_size - 1, x + 1)

        return (x, y)

    def render(self):
        # Build grid representation
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[tuple(self.fire_pos)] = 3  # fire
        grid[tuple(self.drone1)] = 1  # drone 1
        grid[tuple(self.drone2)] = 2 # drone 2

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
        
        return self.fig

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

if __name__ == '__main__':
    env = FireSearchEnv()

    for i in range(15):
        fig = env.render()
        #input("Press enter to close")
        fig.savefig(f"FireEnvStep{i}", dpi=300)
        env.drone1 = env.move(env.drone1, np.random.randint(0, 5))
        env.drone2 = env.move(env.drone2, np.random.randint(0, 5))