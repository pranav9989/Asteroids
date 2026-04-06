import numpy as np
from collections import defaultdict

class QAgent:

    def __init__(self):

        # Only 3 actions: 0=left, 1=right, 2=shoot (removed wait action)
        self.q = defaultdict(lambda: np.zeros(3))

        # Learning parameters
        self.alpha = 0.3  # Increased learning rate (was 0.1)
        self.gamma = 0.95  # Discount factor

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.998  # Slower decay (was 0.995)
        self.epsilon_min = 0.1  # Higher min epsilon (was 0.05)

    def choose_action(self, state):

        # Extract number of asteroids from state
        num_asteroids = 0
        if len(state) >= 3 and state[2] != -1:
            num_asteroids = len(self.asteroids) if hasattr(self, 'asteroids') else 1
        
        if np.random.rand() < self.epsilon:
            # Exploration: bias towards shooting when asteroids exist
            if num_asteroids > 0:
                # 70% shoot, 15% left, 15% right
                rand = np.random.random()
                if rand < 0.7:
                    return 2  # shoot
                elif rand < 0.85:
                    return 0  # left
                else:
                    return 1  # right
            else:
                # No asteroids, only move left/right
                return np.random.randint(2)
        
        # Exploitation: use Q-table
        q_values = self.q[state].copy()
        
        # If no asteroids, disable shooting action
        if num_asteroids == 0:
            q_values[2] = -float('inf')
        
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done=False):

        current_q = self.q[state][action]
        
        if done:
            target = reward
        else:
            best_next = np.max(self.q[next_state])
            target = reward + self.gamma * best_next

        error = target - current_q
        self.q[state][action] += self.alpha * error

    def decay(self):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def set_asteroid_count(self, count):
        """Helper method to pass asteroid count to agent"""
        self.asteroids = [0] * count if hasattr(self, 'asteroids') else None