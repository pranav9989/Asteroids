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

print("="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Reward Structure:")
print(f"  • Destroy Asteroid: +100")
print(f"  • Survival per step: +0.1")
print(f"  • Collision: -100")
print(f"  • 5 asteroids spawn per step!")
print("="*60)

for episode in range(episodes):

    state = env.reset()
    total_reward = 0

    for step in range(max_steps):

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        # Pass done flag to learn method
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay()
    
    # Store statistics
    score_history.append(env.score)
    time_history.append(env.time_alive)
    reward_history.append(total_reward)

    # Print progress every 500 episodes
    if episode % 500 == 0:
        avg_score = np.mean(score_history[-500:]) if episode >= 500 else np.mean(score_history)
        avg_time = np.mean(time_history[-500:]) if episode >= 500 else np.mean(time_history)
        avg_reward = np.mean(reward_history[-500:]) if episode >= 500 else np.mean(reward_history)
        
        print(f"\n📊 Episode: {episode}")
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Score: {env.score}")
        print(f"  Time Alive: {env.time_alive}")
        print(f"  Epsilon: {agent.epsilon:.3f}")
        print(f"  Avg Score (last 500): {avg_score:.2f}")
        print(f"  Avg Time (last 500): {avg_time:.2f}")
        print(f"  Avg Reward (last 500): {avg_reward:.2f}")
        print(f"  Q-table size: {len(agent.q)} states")
        print("-" * 40)
    
    # Early stopping if agent performs well
    if episode > 2000 and np.mean(score_history[-500:]) > 50:
        print(f"\n🎉 Agent mastered the game! Stopping early at episode {episode}")
        print(f"Average Score (last 500): {np.mean(score_history[-500:]):.2f}")
        break

# Save trained model
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(agent.q), f)

# Print final statistics
print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"\n📊 FINAL STATISTICS:")
print(f"  Best Score: {max(score_history)}")
print(f"  Best Survival Time: {max(time_history)}")
print(f"  Final Epsilon: {agent.epsilon:.3f}")
print(f"  Q-table size: {len(agent.q)} states")

print(f"\n📈 LAST 500 EPISODES AVERAGE:")
print(f"  Score: {np.mean(score_history[-500:]):.2f}")
print(f"  Time: {np.mean(time_history[-500:]):.2f}")
print(f"  Reward: {np.mean(reward_history[-500:]):.2f}")

print(f"\n📊 OVERALL AVERAGE:")
print(f"  Score: {np.mean(score_history):.2f}")
print(f"  Time: {np.mean(time_history):.2f}")

# Learning progress summary
print("\n" + "="*60)
print("LEARNING PROGRESS SUMMARY")
print("="*60)

quarters = len(score_history) // 4
if quarters > 0:
    q1_score = np.mean(score_history[:quarters])
    q2_score = np.mean(score_history[quarters:2*quarters])
    q3_score = np.mean(score_history[2*quarters:3*quarters])
    q4_score = np.mean(score_history[3*quarters:])
    
    print(f"\n  Score Progression:")
    print(f"    First quarter:  {q1_score:.2f}")
    print(f"    Second quarter: {q2_score:.2f}")
    print(f"    Third quarter:  {q3_score:.2f}")
    print(f"    Fourth quarter: {q4_score:.2f}")
    
    if q1_score > 0:
        improvement = ((q4_score - q1_score) / q1_score) * 100
        print(f"\n  📈 Overall Improvement: {improvement:.1f}%")
    
    if q4_score > q1_score:
        print("  ✅ Agent IS learning! Scores are improving over time.")
    else:
        print("  ⚠️ Agent may need more episodes.")

print("="*60)

# Save training history
training_data = {
    'scores': score_history,
    'times': time_history,
    'rewards': reward_history,
    'best_score': max(score_history),
    'best_time': max(time_history)
}

with open("training_history.pkl", "wb") as f:
    pickle.dump(training_data, f)
    
print("\n💾 Training history saved to 'training_history.pkl'")
print("🎮 Model saved to 'q_table.pkl'")
print("\n💡 The agent is now trained to prioritize SHOOTING asteroids!")
print("   (100x more reward for shooting than surviving)")
print("="*60)