"""
Environment module for multi-agent fire search simulation
Handles rendering, step execution, and reward computation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from gymnasium import Env
import config as cfg



class SearchEnv(Env):
    """
    Multi-agent search environment with Dec-POMDP framework
    
    Grid representation:
    - White: Empty cell
    - Red: Fire
    - Blue/Green/etc: Drones
    """
    def __init__(self, grid_size=cfg.FIGURE_SIZE):
        self.grid_size = grid_size
        self.fig, self.ax = None, None
        self.fire_pos = np.random.randint(0, self.grid_size, size=2)
        self.patches = []
        
        # Cost parameters
        self.communication_cost = cfg.COMMUNICATION_COST
        self.movement_cost = cfg.MOVEMENT_COST
        
        # Tracking
        self.total_cost = 0.0
        self.fire_extinguished = False
        self.time_to_extinguish = None
        self.total_communications = 0

    def step(self, drones, actions):
        """
        Execute actions for all drones and handle communication
        
        Args:
            drones: list of Drone objects
            actions: list of integer actions, one per drone
            
        Returns:
            tuple: (step_cost, fire_extinguished)
        """
        telemetry_packets = []
        step_cost = 0.0
        
        # Execute all actions and collect telemetry
        for drone, action in zip(drones, actions):
            packet = drone.action(action, self.fire_pos)
            if packet is not None:
                telemetry_packets.append(packet)
                step_cost += self.communication_cost
                self.total_communications += 1
            elif action != 0:
                step_cost += self.movement_cost
        
        # Distribute telemetry packets to all other drones
        for packet in telemetry_packets:
            for drone in drones:
                if drone.drone_id != packet['sender_id']:
                    drone.receive_telemetry(packet, communication_noise=0.05)
        
        # Check if any drone has reached the fire
        for drone in drones:
            if drone.x == self.fire_pos[0] and drone.y == self.fire_pos[1]:
                if not self.fire_extinguished:
                    self.fire_extinguished = True
                    self.time_to_extinguish = drone.time
                    print(f"FIRE EXTINGUISHED by Drone {drone.drone_id}!")
                    print(f"Time: {drone.time:.2f}s")
                    print(f"Total Cost: {self.total_cost + step_cost:.2f}")
                    print(f"Communications: {self.total_communications}")
        
        self.total_cost += step_cost
        return step_cost, self.fire_extinguished

    def render(self, drones):
        """
        Render the current state of the environment
        
        Args:
            drones: list of Drone objects
            
        Returns:
            matplotlib.figure.Figure: the figure object
        """
        # Build grid representation
        grid = np.zeros((self.grid_size, self.grid_size))
        
        if not self.fire_extinguished:
            grid[tuple(self.fire_pos)] = 1  # Fire

        for idx, drone in enumerate(drones):
            grid[tuple(drone.position)] = idx + 2  # Drones

        cmap = colors.ListedColormap(cfg.GRID_COLORS)
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.im = self.ax.imshow(grid, cmap=cmap, norm=norm)
            self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            
            title = (f'Dec-POMDP Multi-Agent Search | Cost: {self.total_cost:.1f} | '
                    f'Comms: {self.total_communications}')
            if self.fire_extinguished:
                title += ' | EXTINGUISHED!'
            self.ax.set_title(title)

            plt.ion()
            plt.show(block=False)
        else:
            self.im.set_data(grid)
            title = (f'Dec-POMDP Multi-Agent Search | Cost: {self.total_cost:.1f} | '
                    f'Comms: {self.total_communications}')
            if self.fire_extinguished:
                title += ' | EXTINGUISHED!'
            self.ax.set_title(title)

        # Clear old patches
        for p in self.patches:
            p.remove()
        self.patches.clear()

        # Draw observation windows for each drone
        for drone in drones:
            corner_x = drone.x - drone.window_size // 2 - 0.5
            corner_y = drone.y - drone.window_size // 2 - 0.5

            rectangle = patches.Rectangle(
                (corner_y, corner_x),
                drone.window_size,
                drone.window_size,
                linewidth=cfg.WINDOW_LINE_WIDTH,
                edgecolor=cfg.WINDOW_EDGE_COLOR,
                facecolor='none'
            )
            self.ax.add_patch(rectangle)
            self.patches.append(rectangle)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return self.fig

    def close(self):
        """Close the rendering window"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None