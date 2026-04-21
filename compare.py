# compare.py (Updated for Epsilon-Greedy Only)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Create assets folder
os.makedirs("assets", exist_ok=True)

def load_results():
    try:
        with open("algorithm_comparison_constant_epsilon.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ Run train.py first!")
        return None

def plot_epsilon_algorithms_comparison(results):
    """Compare Epsilon-Greedy algorithms: Q-Learning vs SARSA vs Double Q"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Epsilon-Greedy Algorithms Comparison (ε=0.1 Constant)\nQ-Learning vs SARSA vs Double Q-Learning', 
                 fontsize=14, fontweight='bold')
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    labels = ['Q-Learning', 'SARSA', 'Double Q-Learning']
    colors = ['green', 'blue', 'purple']
    
    # Plot 1: Average Score
    ax1 = axes[0]
    avg_scores = [results[algo]['avg_score_last_500'] for algo in algorithms]
    best_scores = [results[algo]['best_score'] for algo in algorithms]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_scores, width, label='Avg Score (last 500)', color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, best_scores, width, label='Best Score', color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Score Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Plot 2: Survival Time
    ax2 = axes[1]
    avg_times = [np.mean(results[algo]['times'][-500:]) for algo in algorithms]
    best_times = [results[algo]['best_time'] for algo in algorithms]
    
    bars3 = ax2.bar(x - width/2, avg_times, width, label='Avg Time', color='lightblue', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, best_times, width, label='Best Time', color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel('Survival Time (steps)', fontsize=12)
    ax2.set_title('Survival Time Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Plot 3: Training Time
    ax3 = axes[2]
    train_times = [results[algo]['training_time'] for algo in algorithms]
    
    bars5 = ax3.bar(x, train_times, width, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylabel('Training Time (seconds)', fontsize=12)
    ax3.set_title('Training Speed', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars5, train_times):
        ax3.annotate(f'{time:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('assets/epsilon_algorithms_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_learning_curves(results):
    """Plot learning curves for all algorithms"""
    
    colors = {
        'qlearning': 'green',
        'sarsa': 'blue',
        'double_q': 'purple'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Epsilon-Greedy Algorithms (ε=0.1 Constant)\nQ-Learning vs SARSA vs Double Q-Learning', 
                 fontsize=14, fontweight='bold')
    
    window = 50
    
    # Plot 1: Scores over time
    ax1 = axes[0]
    for name, data in results.items():
        scores = data['scores']
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        episodes = range(window, len(scores) + 1)
        ax1.plot(episodes, smoothed, label=name.upper(), color=colors.get(name, 'gray'), linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Score (Smoothed)', fontsize=12)
    ax1.set_title('Learning Curves - Score', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival Time over time
    ax2 = axes[1]
    for name, data in results.items():
        times = data['times']
        smoothed = np.convolve(times, np.ones(window)/window, mode='valid')
        episodes = range(window, len(times) + 1)
        ax2.plot(episodes, smoothed, label=name.upper(), color=colors.get(name, 'gray'), linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Survival Time (Smoothed)', fontsize=12)
    ax2.set_title('Learning Curves - Survival Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_summary_table(results):
    """Create summary table as a plot"""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    headers = ['Algorithm', 'Avg Score', 'Best Score', 'Avg Time', 'Best Time', 'Train Time']
    
    data = []
    for algo in ['qlearning', 'sarsa', 'double_q']:
        if algo in results:
            d = results[algo]
            algo_name = 'Q-Learning' if algo == 'qlearning' else 'SARSA' if algo == 'sarsa' else 'Double Q'
            data.append([
                algo_name,
                f"{d['avg_score_last_500']:.2f}",
                f"{d['best_score']}",
                f"{np.mean(d['times'][-500:]):.1f}",
                f"{d['best_time']}",
                f"{d['training_time']:.1f}s"
            ])
    
    # Sort by avg score
    data.sort(key=lambda x: float(x[1]), reverse=True)
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color coding
    for i, row in enumerate(data):
        if 'Q-Learning' in row[0]:
            table[(i+1, 0)].set_facecolor('#c8e6c9')
        elif 'SARSA' in row[0]:
            table[(i+1, 0)].set_facecolor('#bbdefb')
        else:
            table[(i+1, 0)].set_facecolor('#f8bbd0')
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#333333')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)
    
    ax.set_title('EPSILON-GREEDY ALGORITHMS COMPARISON (ε=0.1 Constant)\n3000 Episodes Training', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('assets/comparison_summary_table.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_comparison_table(results):
    """Print detailed comparison table in console"""
    
    print("\n" + "="*80)
    print("🏆 EPSILON-GREEDY ALGORITHMS COMPARISON (ε=0.1 Constant) 🏆")
    print("="*80)
    
    print(f"\n{'Algorithm':<20} {'Avg Score':<15} {'Best Score':<15} {'Avg Time':<15} {'Best Time':<15} {'Train Time':<12}")
    print("-"*92)
    
    comparison_data = []
    for algo in ['qlearning', 'sarsa', 'double_q']:
        if algo in results:
            d = results[algo]
            algo_name = 'Q-Learning' if algo == 'qlearning' else 'SARSA' if algo == 'sarsa' else 'Double Q'
            avg_time = np.mean(d['times'][-500:]) if len(d['times']) >= 500 else np.mean(d['times'])
            
            comparison_data.append({
                'algo': algo_name,
                'avg_score': d['avg_score_last_500'],
                'best_score': d['best_score'],
                'avg_time': avg_time,
                'best_time': d['best_time'],
                'train_time': d['training_time']
            })
    
    # Sort by avg score
    comparison_data.sort(key=lambda x: x['avg_score'], reverse=True)
    
    for i, d in enumerate(comparison_data, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {d['algo']:<17} {d['avg_score']:<15.2f} {d['best_score']:<15} {d['avg_time']:<15.1f} {d['best_time']:<15} {d['train_time']:<12.1f}s")
    
    # Best overall
    print("\n" + "="*80)
    print("🏆 BEST IN EACH CATEGORY")
    print("="*80)
    
    best_score = max(comparison_data, key=lambda x: x['avg_score'])
    best_peak = max(comparison_data, key=lambda x: x['best_score'])
    fastest_train = min(comparison_data, key=lambda x: x['train_time'])
    best_survival = max(comparison_data, key=lambda x: x['avg_time'])
    
    print(f"\n  🏆 Best Average Score:     {best_score['algo']} ({best_score['avg_score']:.2f})")
    print(f"  🎯 Best Peak Score:        {best_peak['algo']} ({best_peak['best_score']})")
    print(f"  ⏱️  Best Survival Time:     {best_survival['algo']} ({best_survival['avg_time']:.1f} steps)")
    print(f"  ⚡ Fastest Training:       {fastest_train['algo']} ({fastest_train['train_time']:.1f}s)")
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    
    winner = comparison_data[0]
    print(f"\n  ✅ BEST ALGORITHM: {winner['algo']}")
    print(f"     • Average Score: {winner['avg_score']:.2f}")
    print(f"     • Training Time: {winner['train_time']:.1f} seconds")
    print(f"     • Best Score: {winner['best_score']}")

def main():
    results = load_results()
    if results is None:
        return
    
    print("\n📊 Generating comparison for Epsilon-Greedy algorithms...")
    
    # Print console comparison
    print_detailed_comparison_table(results)
    
    # Generate plots
    plot_epsilon_algorithms_comparison(results)
    plot_learning_curves(results)
    plot_summary_table(results)
    
    print("\n✅ All visualizations saved in 'assets/' folder:")
    print("  📊 assets/epsilon_algorithms_comparison.png")
    print("  📊 assets/learning_curves.png")
    print("  📊 assets/comparison_summary_table.png")

if __name__ == "__main__":
    main()