import pickle
import numpy as np
from env import AsteroidsEnv
from q_agent import QAgent
import matplotlib.pyplot as plt

env = AsteroidsEnv(width=10, height=10)
agent = QAgent()

episodes = 10000
max_steps = 1000

# Track statistics
ai_scores = []
ai_times = []
human_scores = []  # We'll simulate human performance for comparison
human_times = []

# Simulated human performance (for training reference)
HUMAN_SKILL_LEVEL = 0.7  # 70% of optimal play

for episode in range(episodes):

    state = env.reset()
    total_reward = 0

    for step in range(max_steps):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay()
    
    # Store AI statistics
    ai_scores.append(env.score)
    ai_times.append(env.time_alive)
    
    # Simulate human performance (random actions with occasional shooting)
    # This creates a baseline for comparison
    human_score = 0
    human_time = 0
    sim_env = AsteroidsEnv(width=10, height=10)
    sim_env.reset()
    
    for _ in range(max_steps):
        # Human-like behavior: sometimes shoot, sometimes move randomly
        if np.random.random() < HUMAN_SKILL_LEVEL:
            # Smart move: try to shoot if asteroid above
            action = 2  # shoot
        else:
            # Random move
            action = np.random.randint(3)
        
        _, _, sim_done = sim_env.step(action)
        if sim_done:
            break
    
    human_scores.append(sim_env.score)
    human_times.append(sim_env.time_alive)

    if episode % 500 == 0:
        avg_ai_score = np.mean(ai_scores[-500:])
        avg_ai_time = np.mean(ai_times[-500:])
        avg_human_score = np.mean(human_scores[-500:])
        avg_human_time = np.mean(human_times[-500:])
        
        print(f"\nEpisode: {episode}")
        print(f"  AI - Avg Score: {avg_ai_score:.2f}, Avg Time: {avg_ai_time:.2f}")
        print(f"  Human - Avg Score: {avg_human_score:.2f}, Avg Time: {avg_human_time:.2f}")
        print(f"  AI Advantage: +{avg_ai_score - avg_human_score:.2f} points")
        print(f"  Epsilon: {agent.epsilon:.3f}")
        print("-" * 50)

# Save trained model
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(agent.q), f)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ai_scores, label='AI', alpha=0.7)
plt.plot(human_scores, label='Human (simulated)', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('AI vs Human Performance - Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(ai_times, label='AI', alpha=0.7)
plt.plot(human_times, label='Human (simulated)', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Survival Time')
plt.title('AI vs Human Performance - Survival Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('competitive_training_results.png')
plt.show()

print("\n" + "="*50)
print("COMPETITIVE TRAINING COMPLETED")
print("="*50)
print(f"Final AI Avg Score (last 500): {np.mean(ai_scores[-500:]):.2f}")
print(f"Final Human Avg Score (last 500): {np.mean(human_scores[-500:]):.2f}")
print(f"Final AI Avg Time (last 500): {np.mean(ai_times[-500:]):.2f}")
print(f"Final Human Avg Time (last 500): {np.mean(human_times[-500:]):.2f}")
print(f"AI Wins by: {np.mean(ai_scores[-500:]) - np.mean(human_scores[-500:]):.2f} points")
print("="*50)