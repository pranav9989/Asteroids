# train.py (Updated for constant epsilon with rewards tracking)
import pickle
import numpy as np
from env import AsteroidsEnv
from q_agent import QLearningAgent, SARSAAgent, DoubleQLearningAgent
import time

def train_agent(agent_class, agent_name, episodes=3000, max_steps=1000):
    """Train a specific agent with constant epsilon"""
    
    env = AsteroidsEnv()
    agent = agent_class(epsilon=0.15)  # Constant 15% exploration
    
    print(f"\n{'='*60}")
    print(f"TRAINING {agent_name.upper()} AGENT")
    print(f"Epsilon: {agent.epsilon} (CONSTANT - NO DECAY)")
    print(f"{'='*60}")
    
    score_history = []
    time_history = []
    reward_history = []  # NEW: Track rewards per episode
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0  # NEW: Total reward for this episode
        
        # Get initial action
        num_asteroids = len(env.asteroids)
        action = agent.choose_action(state, num_asteroids)
        
        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            episode_reward += reward  # NEW: Accumulate reward
            
            # Get next action for SARSA
            num_asteroids_next = len(env.asteroids)
            next_action = agent.choose_action(next_state, num_asteroids_next) if not done else None
            
            # Learn
            agent.learn(state, action, reward, next_state, done, next_action)
            
            state = next_state
            action = next_action if not done else None
            
            if done:
                break
        
        # No epsilon decay!
        
        # Store statistics
        score_history.append(env.score)
        time_history.append(env.time_alive)
        reward_history.append(episode_reward)  # NEW: Store total reward
        
        # Print progress every 500 episodes
        if episode % 500 == 0 and episode > 0:
            avg_score = np.mean(score_history[-500:])
            avg_time = np.mean(time_history[-500:])
            avg_reward = np.mean(reward_history[-500:])  # NEW: Average reward
            
            print(f"\n📊 Episode: {episode}/{episodes}")
            print(f"  Score: {env.score} (Avg last 500: {avg_score:.2f})")
            print(f"  Time: {env.time_alive} (Avg last 500: {avg_time:.2f})")
            print(f"  Reward: {episode_reward:.1f} (Avg last 500: {avg_reward:.1f})")  # NEW
            print(f"  Epsilon: {agent.epsilon} (CONSTANT)")
    
    training_time = time.time() - start_time
    
    # Save model
    if agent_name == "double_q":
        q_table = agent.get_q_table()
    else:
        q_table = dict(agent.q)
    
    with open(f"q_table_{agent_name}.pkl", "wb") as f:
        pickle.dump(q_table, f)
    
    results = {
        'name': agent_name,
        'scores': score_history,
        'times': time_history,
        'rewards': reward_history,  # NEW: Include rewards in results
        'best_score': max(score_history),
        'best_time': max(time_history),
        'avg_score_last_500': np.mean(score_history[-500:]) if len(score_history) >= 500 else np.mean(score_history),
        'training_time': training_time,
        'q_table_size': len(q_table)
    }
    
    return results

def compare_algorithms():
    """Train and compare all three algorithms"""
    
    algorithms = [
        (QLearningAgent, "qlearning"),
        (SARSAAgent, "sarsa"),
        (DoubleQLearningAgent, "double_q")
    ]
    
    all_results = {}
    
    for agent_class, agent_name in algorithms:
        print(f"\n{'🚀'*40}")
        print(f"Training: {agent_name.upper()}")
        print(f"{'🚀'*40}")
        
        results = train_agent(agent_class, agent_name, episodes=3000)
        all_results[agent_name] = results
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {agent_name.upper()}")
        print(f"{'='*60}")
        print(f"  Best Score: {results['best_score']}")
        print(f"  Avg Score (last 500): {results['avg_score_last_500']:.2f}")
        print(f"  Avg Reward (last 500): {np.mean(results['rewards'][-500:]):.1f}")  # NEW
        print(f"  Training Time: {results['training_time']:.2f} seconds")
        print(f"  Q-table Size: {results['q_table_size']}")
    
    # Save results
    with open("algorithm_comparison_constant_epsilon.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    return all_results

if __name__ == "__main__":
    print("🎮 ASTEROIDS RL - CONSTANT EPSILON (NO DECAY)")
    print("="*60)
    print("3 Algorithms being trained:")
    print("  1. Q-Learning (ε=0.15 constant)")
    print("  2. SARSA (ε=0.15 constant)")
    print("  3. Double Q-Learning (ε=0.15 constant)")
    print("="*60)
    print("\n⚠️ No epsilon decay - constant 15% exploration throughout!")
    print("="*60)
    
    results = compare_algorithms()
    
    print("\n✅ Training Complete!")
    print("Models saved:")
    print("  - q_table_qlearning.pkl")
    print("  - q_table_sarsa.pkl")
    print("  - q_table_double_q.pkl")