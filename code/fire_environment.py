from gymnasium import Env
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches

class Drone():
    """
    |S| = {x, y, fire_found: Bool, time_idx}
    |A| = {stay, move up, move down, move left, move right, communicate}
    H (history) = history of past states
    """
    def __init__(self, grid_size, window_size = 3, time = 0, dt = 0.05):
        self.grid_size = grid_size
        self.window_size = window_size

        self.time = time
        self.dt = dt

        self.position = np.random.randint(0, self.grid_size, size = 2)
        self.fire_found = False

        self.history = [self.state]

    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]

    @property
    def state(self):
        return [self.x, self.y, self.fire_found, self.time]


    def observe(self, fire_pos):
        x_check = (self.x - self.window_size // 2 <= fire_pos[0] <= self.x + self.window_size // 2)
        y_check = (self.y - self.window_size // 2 <= fire_pos[1] <= self.y + self.window_size // 2)
        if x_check and y_check:
            print(f"Found a fire at {self.state} !")
            return True
        else:
            return False


    def action(self, action, fire_pos):
        """
        Triggers either a movement or communication, which in turn causes the drone to observe the fire and update its state history
        action: (int) 0:stay, 1:up, 2:down, 3:left, 4:right, 5: communicate
        fire_pos: (tuple) (x: fire_pos_x, y: fire_pos_y)
        """
        x = self.x
        y = self.y
        if action == 1: y = min(self.grid_size - 1, self.y + 1)
        elif action == 2: y = max(0, self.y - 1)
        elif action == 3: x = max(0, self.x - 1)
        elif action == 4: x = min(self.grid_size - 1, self.x + 1)

        self.position = np.array([x, y])         # Update position
        self.fire_found = self.observe(fire_pos) # Update fire_found
        self.time += self.dt                     # Update time
        self.history.append(self.state)          # Track the state

        return None


class SearchEnv(Env):
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
        self.fig, self.ax = None, None
        self.fire_pos = np.random.randint(0, self.grid_size, size = 2)
        self.patches = []

    def render(self, Drones):
        # Drones: A list of drone objects
        # Build grid representation
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[tuple(self.fire_pos)] = 1  # fire

        for idx, drone in enumerate(Drones):
            grid[tuple(drone.position)] = idx + 2  # drone i 

        cmap = colors.ListedColormap(['white', 'red', 'blue', 'green'])
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

        for p in self.patches:
            p.remove()
        self.patches.clear()

        for drone in Drones:
            corner_x = drone.x - drone.window_size // 2 - 0.5
            corner_y = drone.y - drone.window_size // 2 - 0.5

            rectangle = patches.Rectangle(
                (corner_y, corner_x),
                drone.window_size,
                drone.window_size,
                linewidth = 2,
                edgecolor = 'black',
                facecolor = 'none'
            )
            self.ax.add_patch(rectangle)
            self.patches.append(rectangle)

        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return self.fig

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

if __name__ == '__main__':
    t_0 = 0
    dt = 0.05
    t_f = 5
    N = int((t_f - t_0) / dt)
    print(f"{N} time steps")

    env = SearchEnv()
    grid_size = env.grid_size

    Drone1 = Drone(grid_size, time = t_0)
    while Drone1.observe(env.fire_pos):
            Drone1 = Drone(grid_size, time = t_0) #If the drone observes the fire at t_0, reshuffle the position. 
    
    Drone2 = Drone(grid_size, time = t_0)
    while Drone2.observe(env.fire_pos):
            Drone2 = Drone(grid_size, time = t_0) #If the drone observes the fire at t_0, reshuffle the position. 
    
    for i in range(N):
        fig = env.render([Drone1, Drone2])
        plt.pause(0.2)
        #fig.savefig(f"FireEnvStep{i}", dpi=300)
        Drone1.action(np.random.randint(1, 5), env.fire_pos)
        Drone2.action(np.random.randint(1, 5), env.fire_pos)