"""
Drone module for Dec-POMDP multi-agent system
Implements intelligent decision-making for fire search and extinguish
"""
import numpy as np
from copy import deepcopy
from belief_state import BeliefState
import config as cfg

class Drone():
    """
    Dec-POMDP Agent with belief state and value-based decision making
    """
    def __init__(self, drone_id, grid_size, num_drones, window_size=3, time=0, dt=0.05):
        self.drone_id = drone_id
        self.grid_size = grid_size
        self.window_size = window_size
        self.num_drones = num_drones

        self.time = time
        self.dt = dt

        self.position = np.random.randint(0, self.grid_size, size=2)
        
        # Belief state for fire location
        self.belief_state = BeliefState(grid_size)
        
        self.history = [self.state]
        
        # Track visited cells for exploration bonus
        self.visited_cells = set()
        self.visited_cells.add((self.x, self.y))
        
        # Beliefs about other drones
        self.beliefs = {}
        for i in range(num_drones):
            if i != self.drone_id:
                self.beliefs[i] = {
                    'position': np.array([grid_size // 2, grid_size // 2]),
                    'belief_state': BeliefState(grid_size),
                    'last_update_time': cfg.INITIAL_TIME,
                    'uncertainty': 1.0,
                    'visited_cells': set()  # **NEW: Track other drone's visited cells**
                }
        
        # Dec-POMDP parameters
        self.gamma = cfg.GAMMA  # Discount factor
        self.communication_threshold = cfg.COMMUNICATION_THRESHOLD  # Communicate when uncertainty is high
        self.exploration_bonus = cfg.EXPLORATION_BONUS # Bonus for exploring new cells, promotes drones to move in unexplored directions

    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]

    @property
    def state(self):
        return [self.x, self.y, self.belief_state.fire_found, self.time]

    def observe(self, fire_pos):
        """Update belief based on observation"""
        x_check = (self.x - self.window_size // 2 <= fire_pos[0] <= self.x + self.window_size // 2)
        y_check = (self.y - self.window_size // 2 <= fire_pos[1] <= self.y + self.window_size // 2)
        fire_observed = x_check and y_check
        
        # Update belief state
        self.belief_state.update_with_observation(self.position, self.window_size, fire_observed)
        
        if fire_observed:
            print(f"Drone {self.drone_id} found fire at position {fire_pos}!")
            self.belief_state.fire_location = fire_pos.copy()
        
        return fire_observed

    def create_telemetry_packet(self):
        """Creates telemetry packet with belief state"""
        packet = {
            'sender_id': self.drone_id,
            'timestamp': self.time,
            'position': self.position.copy(),
            'belief_state': deepcopy(self.belief_state),
            'visited_cells': self.visited_cells.copy()
        }
        
        print(f"\n{'='*60}")
        print(f"TELEMETRY SENT by Drone {self.drone_id}")
        print(f"{'='*60}")
        print(f"  Time: {self.time:.2f}s")
        print(f"  Position: ({self.position[0]}, {self.position[1]})")
        print(f"  Fire Found: {self.belief_state.fire_found}")
        print(f"  Belief Entropy: {self.belief_state.get_entropy():.3f}")
        print(f"  Cells Explored: {len(self.visited_cells)}")
        print(f"{'='*60}\n")
        
        return packet

    def receive_telemetry(self, packet, communication_noise=0.1):
        """Receive and merge belief states"""
        sender_id = packet['sender_id']
        
        if sender_id == self.drone_id or sender_id not in self.beliefs:
            return
        
        print(f"\n{'─'*60}")
        print(f"TELEMETRY RECEIVED by Drone {self.drone_id} from Drone {sender_id}")
        print(f"{'─'*60}")
        print(f"  Time: {packet['timestamp']:.2f}s")
        print(f"  Sender Position: ({packet['position'][0]}, {packet['position'][1]})")
        print(f"  Sender Fire Found: {packet['belief_state'].fire_found}")
        
        # Update beliefs about other drone
        old_uncertainty = self.beliefs[sender_id]['uncertainty']
        self.beliefs[sender_id]['position'] = packet['position'].copy()
        self.beliefs[sender_id]['belief_state'] = packet['belief_state']
        self.beliefs[sender_id]['last_update_time'] = packet['timestamp']
        self.beliefs[sender_id]['uncertainty'] = communication_noise
        self.beliefs[sender_id]['visited_cells'] = packet['visited_cells'].copy()

        other_visited = packet['visited_cells']
        num_new_cells = len(other_visited - self.visited_cells)
        self.visited_cells.update(other_visited)
        
        # Merge belief states
        old_entropy = self.belief_state.get_entropy()
        self.belief_state.merge_with_other_belief(packet['belief_state'], weight=0.5)
        new_entropy = self.belief_state.get_entropy()
        
        print(f"  Belief Entropy: {old_entropy:.3f} → {new_entropy:.3f}")
        print(f"  Uncertainty: {old_uncertainty:.2f} → {communication_noise:.2f}")
        print(f"  **Learned {num_new_cells} new visited cells from Drone {sender_id}**")
        print(f"{'─'*60}\n")

    def update_beliefs(self, dt):
        """Update beliefs about other drones over time"""
        for drone_id in self.beliefs:
            time_since_update = self.time - self.beliefs[drone_id]['last_update_time']
            uncertainty_growth_rate = cfg.UNCERTAINTY_GROWTH_RATE
            self.beliefs[drone_id]['uncertainty'] = min(1.0, 
                self.beliefs[drone_id]['uncertainty'] + uncertainty_growth_rate * dt)

    def compute_information_gain(self, next_position):
        """
        Compute expected information gain from moving to next_position
        Based on reduction in belief entropy
        """
        temp_belief = deepcopy(self.belief_state.belief)
        x, y = next_position
        
        # Calculate expected information gain
        observation_area = 0
        for i in range(max(0, x - self.window_size // 2), min(self.grid_size, x + self.window_size // 2 + 1)):
            for j in range(max(0, y - self.window_size // 2), min(self.grid_size, y + self.window_size // 2 + 1)):
                observation_area += temp_belief[i, j]
        
        # Check if ANY drone has visited this cell (including us)
        cell_visited_by_anyone = tuple(next_position) in self.visited_cells
        
        # Double bonus for completely unexplored cells
        exploration_bonus = self.exploration_bonus if not cell_visited_by_anyone else 0
        
        return observation_area + exploration_bonus
    def compute_q_value(self, action):
        """
        Compute Q-value for an action using Dec-POMDP value function
        Q(b, a) = R(b, a) + gamma * V(b')
        
        """
        # Simulate movement actions
        x, y = self.x, self.y
        
        if action == 1:  # Up
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # Down
            y = max(0, y - 1)
        elif action == 3:  # Left
            x = max(0, x - 1)
        elif action == 4:  # Right
            x = min(self.grid_size - 1, x + 1)
        
        next_position = np.array([x, y])
        
        # If fire is found, value is based on distance to fire
        if self.belief_state.fire_found and self.belief_state.fire_location is not None:
            distance_to_fire = np.abs(next_position - self.belief_state.fire_location).sum()
            if distance_to_fire == 0:
                return 100.0  # Huge reward for extinguishing
            else:
                return 10.0 - distance_to_fire
        
        # Otherwise, value is based on information gain
        info_gain = self.compute_information_gain(next_position)
        movement_cost = cfg.MOVEMENT_COST if action != 0 else 0.0
        
        q_value = info_gain - movement_cost - cfg.TIME_COST # Added time cost to hasten our drones. May be erroneous
        
        return q_value

    def should_communicate(self):
        """
        Decide whether to communicate based on:
        1. Belief entropy (uncertainty)
        2. Time since last communication
        3. Whether we found the fire
        """
        # Communicate periodically
        time_step = int(self.time / self.dt)
        
        # If fire found, communicate immediately and then periodically
        if self.belief_state.fire_found:
            if time_step % 10 == 0:  # Every 10 steps after finding fire
                return True
        
        # During search, communicate less frequently (every 30 steps)
        # This allows time for exploration between communications
        if time_step > 0 and time_step % 30 == 0:
            current_entropy = self.belief_state.get_entropy()
            max_entropy = np.log(self.grid_size * self.grid_size)
            normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0
            
            # Only communicate if entropy is still high
            if normalized_entropy > self.communication_threshold:
                return True
        
        return False

    def decide_action_pomdp(self):
        """
        Dec-POMDP decision making using value iteration
        
        Returns:
            int: chosen action (0-5)
        """
        # If at fire location, stay
        if self.belief_state.fire_found and self.belief_state.fire_location is not None:
            if self.x == self.belief_state.fire_location[0] and self.y == self.belief_state.fire_location[1]:
                return 0
        
        # First we dcedie if we should communicate based on entropy/cost tradeoff
        current_entropy = self.belief_state.get_entropy()
        max_entropy = np.log(self.grid_size * self.grid_size)
        normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0
        
        # Communication value: benefit of sharing info vs cost
        comm_value = 5.0 * normalized_entropy - cfg.COMMUNICATION_COST
        
        # Communicate if it has positive value AND we haven't communicated too recently
        time_step = int(self.time / self.dt)
        MIN_COMM_INTERVAL = cfg.MIN_COMM_INTERVAL  # Minimum steps between communications
        
        if comm_value > 0 and time_step % MIN_COMM_INTERVAL == 0:
            if int(self.time / self.dt) % 10 == 0:
                print(f"Drone {self.drone_id} chose action: 5 (communicate), comm_value={comm_value:.2f}")
            return 5
        
        # Then if not communicating, compute Q-values for movement actions only
        q_values = {}
        for action in range(5):  # 0-4: stay and movement
            q_values[action] = self.compute_q_value(action)
        
        # Debug print statement but kinda looks cool when logging in terminal lmao
        if int(self.time / self.dt) % 10 == 0:
            print(f"Drone {self.drone_id} Q-values: {q_values}")
        
        # Select best movement action
        if not self.belief_state.fire_found:
            # During search, exclude stay action
            movement_q_values = {a: q for a, q in q_values.items() if a != 0}
            if movement_q_values:
                best_action = max(movement_q_values, key=movement_q_values.get)
            else:
                best_action = max(q_values, key=q_values.get)
        else:
            # After fire found, can stay if needed
            best_action = max(q_values, key=q_values.get)
        
        if int(self.time / self.dt) % 10 == 0:
            print(f"Drone {self.drone_id} chose action: {best_action} (movement)")
        
        return best_action

    def action(self, action, fire_pos):
        """Execute action and update state"""
        x = self.x
        y = self.y
        
        is_communication = (action == 5)
        
        if action == 1:  # Up
            y = min(self.grid_size - 1, self.y + 1)
        elif action == 2:  # Down
            y = max(0, self.y - 1)
        elif action == 3:  # Left
            x = max(0, self.x - 1)
        elif action == 4:  # Right
            x = min(self.grid_size - 1, self.x + 1)

        self.position = np.array([x, y])
        self.visited_cells.add((x, y))
        self.observe(fire_pos)
        self.time += self.dt
        self.update_beliefs(self.dt)
        self.history.append(self.state)

        telemetry_packet = None
        if is_communication:
            telemetry_packet = self.create_telemetry_packet()

        return telemetry_packet