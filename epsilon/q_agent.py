# q_agent_enhanced.py
import numpy as np
from collections import defaultdict
import random

class BaseAgent:
    """Base class for all RL agents"""
    
    def __init__(self, algorithm="qlearning"):
        self.algorithm = algorithm
        self.actions = 3  # 0=left, 1=right, 2=shoot


class QLearningAgent(BaseAgent):
    """Standard Q-Learning Agent"""
    
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.1):
        super().__init__("qlearning")
        self.q = defaultdict(lambda: np.zeros(3))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def choose_action(self, state, num_asteroids=0):
        if random.random() < self.epsilon:
            # Exploration with bias
            if num_asteroids > 0:
                rand = random.random()
                if rand < 0.7:
                    return 2  # shoot
                elif rand < 0.85:
                    return 0  # left
                else:
                    return 1  # right
            else:
                return random.randint(0, 1)
        
        # Exploitation
        q_values = self.q[state].copy()
        if num_asteroids == 0:
            q_values[2] = -float('inf')
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done, next_action=None):
        current_q = self.q[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state])
        
        self.q[state][action] += self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class SARSAAgent(BaseAgent):
    """SARSA (State-Action-Reward-State-Action) Agent"""
    
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.1):
        super().__init__("sarsa")
        self.q = defaultdict(lambda: np.zeros(3))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def choose_action(self, state, num_asteroids=0):
        if random.random() < self.epsilon:
            if num_asteroids > 0:
                rand = random.random()
                if rand < 0.7:
                    return 2
                elif rand < 0.85:
                    return 0
                else:
                    return 1
            else:
                return random.randint(0, 1)
        
        q_values = self.q[state].copy()
        if num_asteroids == 0:
            q_values[2] = -float('inf')
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done, next_action=None):
        current_q = self.q[state][action]
        
        if done:
            target = reward
        else:
            # SARSA uses the actual next action chosen
            target = reward + self.gamma * self.q[next_state][next_action]
        
        self.q[state][action] += self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DoubleQLearningAgent(BaseAgent):
    """Double Q-Learning Agent (reduces overestimation bias)"""
    
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.1):
        super().__init__("double_q")
        self.q1 = defaultdict(lambda: np.zeros(3))
        self.q2 = defaultdict(lambda: np.zeros(3))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def choose_action(self, state, num_asteroids=0):
        if random.random() < self.epsilon:
            if num_asteroids > 0:
                rand = random.random()
                if rand < 0.7:
                    return 2
                elif rand < 0.85:
                    return 0
                else:
                    return 1
            else:
                return random.randint(0, 1)
        
        # Use average of both Q-tables for action selection
        q_values = (self.q1[state] + self.q2[state]) / 2
        if num_asteroids == 0:
            q_values[2] = -float('inf')
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done, next_action=None):
        if random.random() < 0.5:
            # Update Q1
            current_q = self.q1[state][action]
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q1[next_state])
                target = reward + self.gamma * self.q2[next_state][best_action]
            self.q1[state][action] += self.alpha * (target - current_q)
        else:
            # Update Q2
            current_q = self.q2[state][action]
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q2[next_state])
                target = reward + self.gamma * self.q1[next_state][best_action]
            self.q2[state][action] += self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_table(self):
        """Return average Q-table for saving"""
        return {state: (self.q1[state] + self.q2[state]) / 2 for state in set(self.q1.keys()) | set(self.q2.keys())}