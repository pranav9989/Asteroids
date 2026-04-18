# 🚀 Reinforcement Learning Asteroids Game

## 📌 Overview

This project implements a **Reinforcement Learning (RL) agent** that learns to play a simplified Asteroids game.

The agent controls a spaceship in a 2D grid environment and learns to:

* 🎯 Destroy asteroids (maximize score)
* ⏱️ Survive as long as possible
* ⚡ React to a dynamic and chaotic environment

The agent is trained using **Q-Learning**, a model-free RL algorithm.

---

## 🧠 Key Features

* Multiple asteroids spawning dynamically
* Reward shaping to prioritize shooting
* Real-time visualization using Pygame
* Performance tracking (score, survival time)
* Early stopping during training
* Saved Q-table for inference

---

## 🌍 Environment

### State Space

The environment is modeled as a simplified MDP:

State:

```
(ship_x, nearest_asteroid_x, nearest_asteroid_y)
```

Only the nearest asteroid is considered to reduce state complexity.

---

### Action Space

| Action | Description |
| ------ | ----------- |
| 0      | Move Left   |
| 1      | Move Right  |
| 2      | Shoot       |

---

### Environment Dynamics

* Multiple asteroids fall from the top
* 5 new asteroids are spawned every step
* Asteroids move downward each timestep
* Collision ends the episode

---

### Reward Function

| Event            | Reward |
| ---------------- | ------ |
| Destroy asteroid | +100   |
| Survive step     | +0.1   |
| Collision        | -100   |

This reward design encourages:

* aggressive asteroid destruction
* balanced survival strategy

---

## 🤖 Agent (Q-Learning)

The agent uses **tabular Q-learning**:

Update rule:

```
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```

### Hyperparameters

* Learning rate (α): 0.3
* Discount factor (γ): 0.95
* Epsilon decay: 0.998
* Min epsilon: 0.1

---

### Exploration Strategy

* ε-greedy policy
* Biased exploration:

  * 70% shoot
  * 15% left
  * 15% right

---

## 🏋️ Training

Training is done over multiple episodes:

* Max steps per episode: 1000
* Early stopping if performance is high
* Metrics tracked:

  * score
  * survival time
  * reward

Outputs:

* `q_table.pkl` → trained model
* `training_history.pkl` → analytics

---

## 🎮 Visualization

The trained agent can be visualized using Pygame:

Features:

* Animated asteroids
* Shooting effects
* Score and time display
* Game-over screen with performance rating

---

## 📊 Results

The agent learns to:

* Prioritize shooting over passive survival
* Avoid collisions dynamically
* Adapt to increasing asteroid density

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pygame

---

## 🎯 Conclusion

This project demonstrates how reinforcement learning can be applied to a dynamic game environment, showcasing:

* Sequential decision making
* Reward-driven learning
* Exploration vs exploitation tradeoff

---
