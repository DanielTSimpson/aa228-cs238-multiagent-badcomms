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
from reward_function import global_reward


class SearchEnv(Env):
    """Multi-agent search environment with Dec-POMDP framework"""
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.fig, self.ax = None, None
        self.fire_pos = np.random.randint(0, self.grid_size, size=2)
        self.patches = []
        self.communication_cost = cfg.COMMUNICATION_COST
        self.movement_cost = cfg.MOVEMENT_COST
        self.time_cost = cfg.TIME_COST
        self.total_cost = 0.0
        self.fire_extinguished = False
        self.time_to_extinguish = None
        self.total_communications = 0

        # Per-step time penalty parameter for global reward function
        self.kappa = cfg.TIME_COST

    def step(self, drones, actions):
        """Execute actions, handle communication, and compute global reward."""
        telemetry_packets = []
        step_movement_cost = 0.0
        comm_count = 0

        # 0) Team belief BEFORE actions
        prev_belief = self._get_team_belief(drones)

        # 1) Execute each drone action (movement, local observation, time update)
        for drone, action in zip(drones, actions):
            packet = drone.action(action, self.fire_pos)

            if packet is not None:
                telemetry_packets.append(packet)
                comm_count += 1
            elif action != 0:
                step_movement_cost += self.movement_cost

        # 2) Handle communication (merge beliefs)
        for packet in telemetry_packets:
            for drone in drones:
                if drone.drone_id != packet['sender_id']:
                    drone.receive_telemetry(packet, communication_noise=0.05)

        # 3) Check for fire extinguished (unchanged)
        for drone in drones:
            if drone.x == self.fire_pos[0] and drone.y == self.fire_pos[1]:
                if not self.fire_extinguished:
                    self.fire_extinguished = True
                    self.time_to_extinguish = drone.time
                    total_step_cost = (
                        step_movement_cost + self.communication_cost * comm_count + self.time_cost
                    )
                    print(f"FIRE EXTINGUISHED by Drone {drone.drone_id}!")
                    print(f"Time: {drone.time:.2f}s")
                    print(
                        f"Total Cost: {self.total_cost + total_step_cost:.2f}"
                    )
                    print(
                        f"Communications: {self.total_communications + comm_count}"
                    )
                    print(
                        f"Time: {self.time_cost}"
                    )

        # Accumulate “real” costs for logging
        self.total_communications += comm_count
        self.total_cost += step_movement_cost + self.communication_cost * comm_count + self.time_cost

        # 4) Team belief AFTER actions + communication
        next_belief = self._get_team_belief(drones)

        # 5) Global reward from reward_function module
        communicated = comm_count > 0

        # Fold movement cost into the effective kappa for this step
        effective_kappa = self.kappa + step_movement_cost

        reward = global_reward(
            prev_belief,
            next_belief,
            kappa=effective_kappa,
            comm_cost=self.communication_cost,
            communicated=communicated,
        )

        return reward, self.fire_extinguished

    def render(self, drones):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        if not self.fire_extinguished:
            grid[tuple(self.fire_pos)] = 1

        for idx, drone in enumerate(drones):
            grid[tuple(drone.position)] = idx + 2

        cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'orange', 'purple'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.im = self.ax.imshow(grid, cmap=cmap, norm=norm)
            self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            
            title = f'Dec-POMDP Multi-Agent Search | Cost: {self.total_cost:.1f} | Comms: {self.total_communications}'
            if self.fire_extinguished:
                title += ' | EXTINGUISHED!'
            self.ax.set_title(title)

            plt.ion()
            plt.show(block=False)
        else:
            self.im.set_data(grid)
            title = f'Dec-POMDP Multi-Agent Search | Cost: {self.total_cost:.1f} | Comms: {self.total_communications}'
            if self.fire_extinguished:
                title += ' | EXTINGUISHED!'
            self.ax.set_title(title)

        for p in self.patches:
            p.remove()
        self.patches.clear()

        for drone in drones:
            corner_x = drone.x - drone.window_size // 2 - 0.5
            corner_y = drone.y - drone.window_size // 2 - 0.5

            rectangle = patches.Rectangle(
                (corner_y, corner_x),
                drone.window_size,
                drone.window_size,
                linewidth=2,
                edgecolor='black',
                facecolor='none'
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


    # Helper function to get the team-level belief over fire location for global reward function
    def _get_team_belief(self, drones):
        """
        Construct a team-level belief over fire location by averaging
        individual drones' beliefs and normalizing.
        Returns a 1D probability vector.
        """
        beliefs = [d.belief_state.belief for d in drones]
        stacked = np.stack(beliefs, axis=0)            # shape: (num_drones, G, G)
        team_belief = stacked.mean(axis=0)             # average over drones

        # Normalize just in case of numerical drift
        total = team_belief.sum()
        if total > 0:
            team_belief = team_belief / total
        else:
            # fallback: uniform if something went horribly wrong
            team_belief = np.ones_like(team_belief) / team_belief.size

        return team_belief.ravel()