"""
Main simulation script for Dec-POMDP multi-agent fire search
Run this file to start the simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from environment import SearchEnv
import config as cfg


def initialize_drones(num_drones, grid_size, env, t_0=0, obs_window_size=3):
    """
    Initialize drones at random positions that don't see the fire initially
    
    Args:
        num_drones: number of drones to create
        grid_size: size of the grid
        env: SearchEnv object
        t_0: initial time
        
    Returns:
        list: list of Drone objects
    """
    drones = []
    for drone_id in range(num_drones):
        drone = Drone(drone_id=drone_id, grid_size=grid_size, 
                     num_drones=num_drones, time=t_0, window_size=obs_window_size)
        # Reshuffle if drone starts at the same loc as fire
        while drone.observe(env.fire_pos):
            drone = Drone(drone_id=drone_id, grid_size=grid_size, 
                         num_drones=num_drones, time=t_0, window_size=obs_window_size)
        drones.append(drone)
    return drones


def print_initial_config(env, drones):
    """Print initial configuration information"""
    print(f"\n{'='*60}")
    print(f"INITIAL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Fire location: {env.fire_pos}")
    for drone in drones:
        print(f"Drone {drone.drone_id} starts at: ({drone.x}, {drone.y})")
    print(f"{'='*60}\n")


def print_periodic_status(i, drones, grid_size):
    """Print periodic status updates"""
    print(f"\n--- Time step {i} (t={drones[0].time:.2f}s) ---")
    for drone in drones:
        entropy = drone.belief_state.get_entropy()
        explored_pct = len(drone.visited_cells) / (grid_size * grid_size) * 100
        print(f"Drone {drone.drone_id}: Pos ({drone.x}, {drone.y}), "
              f"Entropy: {entropy:.3f}, "
              f"Fire Found: {drone.belief_state.fire_found}, "
              f"Explored: {explored_pct:.1f}%")


def print_final_results(env):
    """Print final simulation results"""
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE - Dec-POMDP Performance")
    print(f"{'='*60}")
    print(f"Total cost: {env.total_cost:.2f}")
    print(f"Total communications: {env.total_communications}")
    if env.fire_extinguished:
        print(f"Fire extinguished at time: {env.time_to_extinguish:.2f}s")
        print(f"SUCCESS!")
    else:
        print(f"Fire NOT extinguished within time limit")
        print(f"FAILED!")
    print(f"{'='*60}")


def run_simulation(grid_size=10, num_drones=2, t_f=10, dt=0.05, 
                   status_interval=20, render_pause=0.1):
    """
    Run the complete Dec-POMDP multi-agent simulation
    
    Args:
        grid_size: size of the grid (NxN)
        num_drones: number of drones
        t_f: maximum simulation time
        dt: time step size
        status_interval: steps between status updates
        render_pause: pause duration for rendering (seconds)
    """
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    window_size = cfg.OBSERVATION_WINDOW_SIZE
    N = int((t_f - t_0) / dt)
    print(f"Max {N} time steps (Dec-POMDP with Value Iteration)")

    # Initialize environment and drones
    env = SearchEnv(grid_size=grid_size)
    drones = initialize_drones(num_drones, grid_size, env, t_0, obs_window_size=window_size)
    
    print_initial_config(env, drones)
    
    # Main simulation loop
    for i in range(N):
        # Render current state
        fig = env.render(drones)
        plt.pause(render_pause)
        
        # Check if fire is extinguished
        if env.fire_extinguished:
            print(f"Fire extinguished! Showing final state...")
            for j in range(10):  # Show final state for 10 frames
                fig = env.render(drones)
                plt.pause(0.2)
            break
        
        # Dec-POMDP decision making for each drone
        actions = []
        for drone in drones:
            action = drone.decide_action_pomdp()
            actions.append(action)
        
        # Execute actions
        step_cost, fire_out = env.step(drones, actions)
        
        # Print periodic status updates
        if i % status_interval == 0:
            print_periodic_status(i, drones, grid_size)
    
    # Print final results
    print_final_results(env)
    
    env.close()


if __name__ == '__main__':
    run_simulation(
        grid_size=cfg.GRID_SIZE,
        num_drones=cfg.NUM_DRONES,
        t_f=cfg.MAX_SIMULATION_TIME,
        dt=cfg.TIME_STEP,
        status_interval=cfg.STATUS_UPDATE_INTERVAL,
        render_pause=cfg.RENDER_PAUSE
    )