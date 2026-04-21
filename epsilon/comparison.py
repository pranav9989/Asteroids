# visualize_comparison.py (Updated for 3 algorithms)
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_comparison_results():
    try:
        with open("algorithm_comparison.pkl", "rb") as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        print("❌ No comparison results found. Run train_all_algorithms.py first.")
        return None

def plot_learning_curves(results):
    colors = {
        'qlearning': 'green',
        'sarsa': 'blue',
        'double_q': 'purple'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('3 Algorithm Comparison: Q-Learning vs SARSA vs Double Q-Learning', fontsize=14, fontweight='bold')
    
    # Plot 1: Scores
    ax1 = axes[0]
    for name, data in results.items():
        scores = data['scores']
        window = 50
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        episodes = range(window, len(scores) + 1)
        ax1.plot(episodes, smoothed, label=name.upper(), color=colors.get(name, 'gray'), linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Learning Curves - Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival Time
    ax2 = axes[1]
    for name, data in results.items():
        times = data['times']
        window = 50
        smoothed = np.convolve(times, np.ones(window)/window, mode='valid')
        episodes = range(window, len(times) + 1)
        ax2.plot(episodes, smoothed, label=name.upper(), color=colors.get(name, 'gray'), linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Survival Time')
    ax2.set_title('Learning Curves - Survival Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bar chart
    ax3 = axes[2]
    algorithms = list(results.keys())
    avg_scores = [results[algo]['avg_score_last_500'] for algo in algorithms]
    best_scores = [results[algo]['best_score'] for algo in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax3.bar(x - width/2, avg_scores, width, label='Avg Score (last 500)', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, best_scores, width, label='Best Score', color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Score')
    ax3.set_title('Final Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels([algo.upper() for algo in algorithms])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    results = load_comparison_results()
    if results is None:
        return
    
    print("\n📊 Generating comparison visualizations...")
    
    print("\n" + "="*60)
    print("FINAL COMPARISON - 3 ALGORITHMS")
    print("="*60)
    
    for name, data in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Best Score: {data['best_score']}")
        print(f"  Avg Score (last 500): {data['avg_score_last_500']:.2f}")
        print(f"  Training Time: {data['training_time']:.1f} seconds")
    
    plot_learning_curves(results)
    
    print("\n✅ Visualization saved as 'algorithm_comparison.png'")

if __name__ == "__main__":
    main()