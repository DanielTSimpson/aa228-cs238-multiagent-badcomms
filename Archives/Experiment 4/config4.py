"""
Configuration file for Dec-POMDP multi-agent simulation
Centralized place for all simulation parameters
"""

# Environment parameters
GRID_SIZE = 15  # Size of the NxN grid

# Drone parameters
NUM_DRONES = 2
OBSERVATION_WINDOW_SIZE = 3
INITIAL_TIME = 0.0
TIME_STEP = 0.05

# Simulation parameters
MAX_SIMULATION_TIME = 20.0
STATUS_UPDATE_INTERVAL = 20
RENDER_PAUSE = 0.1

# Dec-POMDP parameters
GAMMA = 0.95
COMMUNICATION_THRESHOLD = 0.2
EXPLORATION_BONUS = 2.0  # Bonus reward for exploring new cells, promotes active exploration of new cells

# Cost parameters
COMMUNICATION_COST = 2.0
MOVEMENT_COST = 1.0
# (UNUSED) STAY_COST = 0.0
TIME_COST = 0.0 # AKA KAPPA

# Communication parameters
COMMUNICATION_NOISE = 0.05
UNCERTAINTY_GROWTH_RATE = 0.1
MIN_COMM_INTERVAL = 5 # Can change this, basically is a threshold that prevents drones from just spam communicating

# Rendering parameters
FIGURE_SIZE = (7, 7)  # Size of the matplotlib figure
GRID_COLORS = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # Cell colors
WINDOW_EDGE_COLOR = 'black'
WINDOW_LINE_WIDTH = 3


def get_environment_config():
    """Returns dictionary of environment configuration"""
    return {
        'grid_size': GRID_SIZE,
        'communication_cost': COMMUNICATION_COST,
        'movement_cost': MOVEMENT_COST,
    }


def get_drone_config():
    """Returns dictionary of drone configuration"""
    return {
        'window_size': OBSERVATION_WINDOW_SIZE,
        'dt': TIME_STEP,
        'gamma': GAMMA,
        'communication_threshold': COMMUNICATION_THRESHOLD,
        'exploration_bonus': EXPLORATION_BONUS,
    }


def get_simulation_config():
    """Returns dictionary of simulation configuration"""
    return {
        'num_drones': NUM_DRONES,
        'max_time': MAX_SIMULATION_TIME,
        'dt': TIME_STEP,
        'status_interval': STATUS_UPDATE_INTERVAL,
        'render_pause': RENDER_PAUSE,
    }