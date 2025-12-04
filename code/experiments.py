from drone import Drone
from environment import SearchEnv
import config as cfg
import matplotlib.pyplot as plt
import numpy as np


def print_initial_config(env, drones):
    """Print initial configuration information"""
    print(f"\n{'='*60}")
    print(f"INITIAL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Fire location: {env.fire_pos}")
    for drone in drones:
        print(f"Drone {drone.drone_id} starts at: ({drone.x}, {drone.y})")
    print(f"{'='*60}\n")


def print_periodic_status(i, drones, reward, grid_size):
    """Print periodic status updates"""
    print(f"\n--- Time step {i} (t={drones[0].time:.2f}s) ---")
    print(f"Reward this step: {reward:.3f}")
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
        print(f"SUCCESS! ✓")
    else:
        print(f"Fire NOT extinguished within time limit")
        print(f"FAILED ✗")
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
    N = int((t_f - t_0) / dt)
    print(f"Max {N} time steps (Dec-POMDP with Value Iteration)")

    env = SearchEnv(grid_size=grid_size)
    env.fire_pos = np.array([grid_size - 2, grid_size - 2])
    num_drones = cfg.NUM_DRONES

    Drone1 = Drone(drone_id=0, grid_size=grid_size, num_drones=num_drones, time=t_0)
    Drone1.position = np.array([1, 1])
    
    Drone2 = Drone(drone_id=1, grid_size=grid_size, num_drones=num_drones, time=t_0)
    Drone2.position = np.array([1, grid_size - 2])

    drones = [Drone1, Drone2]
    
    print_initial_config(env, drones)
    
    fig = env.render(drones)
    plt.savefig("InitialPositions.png")
    for i in range(N):
        fig = env.render(drones)
        plt.pause(0.1)
        
        if env.fire_extinguished:
            print(f"Fire extinguished! Showing final state...")
            for j in range(10):
                fig = env.render(drones)
                plt.pause(0.2)
            break
        
        # Dec-POMDP decision making
        actions = []
        for drone in drones:
            action = drone.decide_action_pomdp()
            actions.append(action)
        
        reward, fire_out = env.step(drones, actions)

        # Print periodic status updates
        if i % status_interval == 0:
            print_periodic_status(i, drones, reward, grid_size)
    
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
    

