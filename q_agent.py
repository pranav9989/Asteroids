import numpy as np
from collections import defaultdict

class QAgent:

    def __init__(self):

        self.q = defaultdict(lambda: np.zeros(4))

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(4)

        return np.argmax(self.q[state])

    def learn(self, state, action, reward, next_state):

        best_next = np.max(self.q[next_state])

        target = reward + self.gamma * best_next

        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def decay(self):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)