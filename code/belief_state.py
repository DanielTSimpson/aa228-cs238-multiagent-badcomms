"""
Belief State module for Dec-POMDP framework
Represents probability distributions over fire locations
"""
import numpy as np
from copy import deepcopy
import config as cfg
from reward_function import compute_entropy

class BeliefState:
    """
    Represents a probability distribution over possible fire locations
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # Uniform prior
        self.belief = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
        self.fire_found = False
        self.fire_location = None
    
    def update_with_observation(self, drone_position, window_size, fire_observed):
        """
        Bayesian update based on drone observation
        """
        if fire_observed:
            self.belief = np.zeros((self.grid_size, self.grid_size))
            x, y = drone_position
            for i in range(max(0, x - window_size // 2), min(self.grid_size, x + window_size // 2 + 1)):
                for j in range(max(0, y - window_size // 2), min(self.grid_size, y + window_size // 2 + 1)):
                    self.belief[i, j] = 1.0
            self.belief /= self.belief.sum()
            self.fire_found = True
        else:
            # Fire not found - zero out probability in observed area
            x, y = drone_position
            for i in range(max(0, x - window_size // 2), min(self.grid_size, x + window_size // 2 + 1)):
                for j in range(max(0, y - window_size // 2), min(self.grid_size, y + window_size // 2 + 1)):
                    self.belief[i, j] = 0.0

            if self.belief.sum() > 0:
                self.belief /= self.belief.sum()
    
    def get_entropy(self):
        """Calculate entropy of belief distribution (measure of uncertainty)"""
        flat_belief = self.belief.flatten()
        flat_belief = flat_belief[flat_belief > 0] 
        if len(flat_belief) == 0:
            return 0.0
        
        # belief without compute_entropy function:
        #return -np.sum(flat_belief * np.log(flat_belief + 1e-10))

        return compute_entropy(self.belief)
    
    def get_most_likely_location(self):
        """Return the cell with highest probability"""
        max_idx = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return np.array(max_idx)
    
    def merge_with_other_belief(self, other_belief, weight=0.5):
        """
        Merge this belief with another drone's belief
        Uses weighted average
        """
        if other_belief.fire_found:
            # If other drone found fire, adopt their belief
            self.belief = other_belief.belief.copy()
            self.fire_found = True
            self.fire_location = other_belief.fire_location
        else:
            # Merge beliefs using weighted average
            self.belief = weight * self.belief + (1 - weight) * other_belief.belief
            if self.belief.sum() > 0:
                self.belief /= self.belief.sum()