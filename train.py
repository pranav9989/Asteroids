import pickle
import numpy as np
from env import AsteroidsEnv
from q_agent import QAgent

env = AsteroidsEnv()
agent = QAgent()

episodes = 5000
max_steps = 1000

# Track statistics for analysis
score_history = []
time_history = []
reward_history = []

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
    
    # Store statistics
    score_history.append(env.score)
    time_history.append(env.time_alive)
    reward_history.append(total_reward)

    if episode % 500 == 0:
        avg_score = np.mean(score_history[-500:]) if episode >= 500 else np.mean(score_history)
        avg_time = np.mean(time_history[-500:]) if episode >= 500 else np.mean(time_history)
        print(f"Episode: {episode}")
        print(f"  Reward: {total_reward:.1f}")
        print(f"  Score: {env.score}")
        print(f"  Time Alive: {env.time_alive}")
        print(f"  Epsilon: {agent.epsilon:.3f}")
        print(f"  Avg Score (last 500): {avg_score:.2f}")
        print(f"  Avg Time (last 500): {avg_time:.2f}")
        print("-" * 40)

# save trained model
with open("q_table.pkl","wb") as f:
    pickle.dump(dict(agent.q), f)

# Print final statistics
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)
print(f"Final Avg Score (last 500 episodes): {np.mean(score_history[-500:]):.2f}")
print(f"Final Avg Time (last 500 episodes): {np.mean(time_history[-500:]):.2f}")
print(f"Best Score: {max(score_history)}")
print(f"Best Time: {max(time_history)}")
print(f"Q-table size: {len(agent.q)} states")
print("="*50)