from drone import Drone
from environment import SearchEnv
import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import csv

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
    window_size = cfg.OBSERVATION_WINDOW_SIZE
    N = int((t_f - t_0) / dt)
    print(f"Max {N} time steps (Dec-POMDP with Value Iteration)")

    entropy_drone1 = []
    entropy_drone2 = []
    time = []

    env = SearchEnv(grid_size=grid_size)
    env.fire_pos = np.array([grid_size - 2, grid_size - 2])
    num_drones = cfg.NUM_DRONES

    Drone1 = Drone(drone_id=0, grid_size=grid_size, num_drones=num_drones, time=t_0, window_size=window_size)
    Drone1.position = np.array([1, 1])
    
    Drone2 = Drone(drone_id=1, grid_size=grid_size, num_drones=num_drones, time=t_0, window_size=window_size)
    Drone2.position = np.array([1, grid_size - 2])

    drones = [Drone1, Drone2]
    
    print_initial_config(env, drones)
    
    #fig = env.render(drones)
    #plt.savefig("InitialPositions.png")
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

        #Update metric trackings
        entropy_drone1.append(Drone1.belief_state.get_entropy())
        entropy_drone2.append(Drone2.belief_state.get_entropy())
        time.append(dt * i)

    # Print final results
    print_final_results(env)

    #Update metrics
    final_time = env.time_to_extinguish
    total_cost = env.total_cost
    total_comms = env.total_communications

    env.close()

    return entropy_drone1, entropy_drone2, time, final_time, total_cost, total_comms


def plot_results(entropy1, entropy2, time, final_time, total_cost, total_comms, N, filename_prefix):
    """
    Plots the entropy of both drones over time and displays simulation metrics.
    entropy1: The entropy as a 1D list of Drone 1 over the entire time interval
    entropy1: The entropy as a 1D list of Drone 1 over the entire time interval
    time: The time interval as a 1D list
    final_time: The time it took the drones to extinguish the fire
    total_cost: The total cost of the drone's set of actions
    total_comms: The total number of communication actions the drones took
    N: Simulation index 
    """
    # 1. Turn OFF interactive mode so plt.show() blocks and keeps the window open
    plt.ioff() 

    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(time, entropy1, label='Drone 1 Entropy', linewidth=2, color='blue')
    plt.plot(time, entropy2, label='Drone 2 Entropy', linewidth=2, color='orange')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Belief Entropy (Uncertainty)', fontsize=12)
    plt.title('Multi-Agent Dec-POMDP: Belief Entropy Over Time', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    time_str = f"{final_time:.2f}s" if final_time is not None else "Failed"
    stats_text = (
        f"RESULTS SUMMARY\n"
        f"----------------\n"
        f"Extinguish Time: {time_str}\n"
        f"Total Cost:      {total_cost:.2f}\n"
        f"# of Comm Actions:    {total_comms}"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='bottom', bbox=props, family='monospace')

    plt.tight_layout()
    
    #plt.show(block=True)
    plt.savefig(f"ConfigResults_{N}.png")
    plt.close(fig)



if __name__ == '__main__':
    max_N = 30
    filename_prefix = "Config_"
    csv_filename = filename_prefix + 'results.csv'
    fieldnames = ['Trial #', 'Total Time', '# Comms', 'Total Cost']

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for i in range(max_N):
        entropy_drone1, entropy_drone2, time, final_time, total_cost, total_comms = run_simulation(
            grid_size=cfg.GRID_SIZE,
            num_drones=cfg.NUM_DRONES,
            t_f=cfg.MAX_SIMULATION_TIME,
            dt=cfg.TIME_STEP,
            status_interval=cfg.STATUS_UPDATE_INTERVAL,
            render_pause=cfg.RENDER_PAUSE
        )

        plot_results(entropy_drone1, entropy_drone2, time, final_time, total_cost, total_comms, i, filename_prefix)

        if final_time is not None:
            time_val = round(final_time, 2)
        else:
            time_val = "FAILED"
            
        cost_val = round(total_cost, 2)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({
                'Trial #': i, 
                'Total Time': time_val, 
                '# Comms': total_comms, 
                'Total Cost': cost_val
            })
            
        print(f"Trial {i} saved to {csv_filename}")