# train_all_algorithms.py
import pickle
import numpy as np
from env import AsteroidsEnv
from q_agent import QLearningAgent, SARSAAgent, DoubleQLearningAgent
import time

def train_agent(agent_class, agent_name, episodes=5000, max_steps=1000):
    """Train a specific agent and return results"""
    
    env = AsteroidsEnv()
    agent = agent_class()
    
    print(f"\n{'='*60}")
    print(f"TRAINING {agent_name.upper()} AGENT")
    print(f"{'='*60}")
    
    score_history = []
    time_history = []
    reward_history = []
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        # Get initial action for SARSA
        num_asteroids = len(env.asteroids)
        action = agent.choose_action(state, num_asteroids)
        
        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            
            # Get next action (needed for SARSA)
            num_asteroids_next = len(env.asteroids)
            next_action = agent.choose_action(next_state, num_asteroids_next) if not done else None
            
            # Learn
            agent.learn(state, action, reward, next_state, done, next_action)
            
            state = next_state
            action = next_action if not done else None
            total_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Store statistics
        score_history.append(env.score)
        time_history.append(env.time_alive)
        reward_history.append(total_reward)
        
        # Print progress every 500 episodes
        if episode % 500 == 0 and episode > 0:
            avg_score = np.mean(score_history[-500:])
            avg_time = np.mean(time_history[-500:])
            
            print(f"\n📊 Episode: {episode}")
            print(f"  Score: {env.score} (Avg: {avg_score:.2f})")
            print(f"  Time: {env.time_alive} (Avg: {avg_time:.2f})")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
            # Early stopping
            if avg_score > 50 and episode > 2000:
                print(f"\n🎉 {agent_name} mastered the game! Stopping early")
                break
    
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
        'rewards': reward_history,
        'best_score': max(score_history),
        'best_time': max(time_history),
        'avg_score_last_500': np.mean(score_history[-500:]) if len(score_history) >= 500 else np.mean(score_history),
        'avg_time_last_500': np.mean(time_history[-500:]) if len(time_history) >= 500 else np.mean(time_history),
        'training_time': training_time,
        'final_epsilon': agent.epsilon,
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
        print(f"\n{'🚀'*30}")
        print(f"Starting training for {agent_name.upper()}")
        print(f"{'🚀'*30}")
        
        results = train_agent(agent_class, agent_name, episodes=5000)
        all_results[agent_name] = results
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"{agent_name.upper()} - FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"  Best Score: {results['best_score']}")
        print(f"  Best Time: {results['best_time']}")
        print(f"  Avg Score (last 500): {results['avg_score_last_500']:.2f}")
        print(f"  Training Time: {results['training_time']:.2f} seconds")
        print(f"  Q-table Size: {results['q_table_size']}")
    
    # Save results
    with open("algorithm_comparison.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Print comparison
    print_algorithm_comparison(all_results)
    
    return all_results

def print_algorithm_comparison(results):
    """Print comparison table"""
    
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*70)
    print(f"{'Algorithm':<15} {'Best Score':<12} {'Avg Score':<12} {'Best Time':<12} {'Training Time':<15}")
    print("-"*70)
    
    for name, data in results.items():
        print(f"{name:<15} {data['best_score']:<12} {data['avg_score_last_500']:<12.2f} "
              f"{data['best_time']:<12} {data['training_time']:<15.1f}")
    
    print("="*70)
    
    # Find best
    best_avg = max(results.items(), key=lambda x: x[1]['avg_score_last_500'])
    best_score = max(results.items(), key=lambda x: x[1]['best_score'])
    
    print(f"\n🏆 Best Average Score: {best_avg[0].upper()} ({best_avg[1]['avg_score_last_500']:.2f})")
    print(f"🏆 Best Peak Performance: {best_score[0].upper()} ({best_score[1]['best_score']})")

if __name__ == "__main__":
    print("🎮 ASTEROIDS RL - 3 ALGORITHM COMPARISON")
    print("Comparing: Q-Learning, SARSA, Double Q-Learning")
    print("\nNote: Each algorithm will train for up to 5000 episodes")
    
    results = compare_algorithms()
    
    print("\n✅ Training Complete!")
    print("Models saved:")
    print("  - q_table_qlearning.pkl")
    print("  - q_table_sarsa.pkl")
    print("  - q_table_double_q.pkl")