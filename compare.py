# compare.py (Updated - Each graph saved separately)
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

def plot_score_comparison(results):
    """Plot 1: Score Comparison Bar Chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    labels = ['Q-Learning', 'SARSA', 'Double Q-Learning']
    
    avg_scores = [results[algo]['avg_score_last_500'] for algo in algorithms]
    best_scores = [results[algo]['best_score'] for algo in algorithms]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_scores, width, label='Avg Score (last 500)', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, best_scores, width, label='Best Score', 
                   color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Comparison (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('assets/score_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/score_comparison.png")

def plot_survival_comparison(results):
    """Plot 2: Survival Time Comparison Bar Chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    labels = ['Q-Learning', 'SARSA', 'Double Q-Learning']
    
    avg_times = [np.mean(results[algo]['times'][-500:]) for algo in algorithms]
    best_times = [results[algo]['best_time'] for algo in algorithms]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_times, width, label='Avg Time', 
                   color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, best_times, width, label='Best Time', 
                   color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Survival Time (steps)', fontsize=12)
    ax.set_title('Survival Time Comparison (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('assets/survival_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/survival_comparison.png")

def plot_training_time_comparison(results):
    """Plot 3: Training Time Comparison Bar Chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    labels = ['Q-Learning', 'SARSA', 'Double Q-Learning']
    colors = ['green', 'blue', 'purple']
    
    train_times = [results[algo]['training_time'] for algo in algorithms]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, train_times, width=0.5, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Speed Comparison (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, train_times):
        ax.annotate(f'{time:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('assets/training_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/training_time_comparison.png")

def plot_reward_comparison(results):
    """Plot 4: Average Reward Comparison Bar Chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    labels = ['Q-Learning', 'SARSA', 'Double Q-Learning']
    
    avg_rewards = [np.mean(results[algo]['rewards'][-500:]) for algo in algorithms]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_rewards, width=0.5, color=['skyblue', 'lightblue', 'lightcoral'], 
                  alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Average Reward per Episode', fontsize=12)
    ax.set_title('Reward Comparison (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, reward in zip(bars, avg_rewards):
        ax.annotate(f'{reward:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('assets/reward_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/reward_comparison.png")

def plot_score_learning_curves(results):
    """Plot 5: Score Learning Curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'qlearning': 'green', 'sarsa': 'blue', 'double_q': 'purple'}
    window = 50
    
    for name, data in results.items():
        scores = data['scores']
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        episodes = range(window, len(scores) + 1)
        label = 'Q-Learning' if name == 'qlearning' else 'SARSA' if name == 'sarsa' else 'Double Q'
        ax.plot(episodes, smoothed, label=label, color=colors.get(name, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Smoothed)', fontsize=12)
    ax.set_title('Score Learning Curves (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/score_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/score_learning_curves.png")

def plot_survival_learning_curves(results):
    """Plot 6: Survival Time Learning Curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'qlearning': 'green', 'sarsa': 'blue', 'double_q': 'purple'}
    window = 50
    
    for name, data in results.items():
        times = data['times']
        smoothed = np.convolve(times, np.ones(window)/window, mode='valid')
        episodes = range(window, len(times) + 1)
        label = 'Q-Learning' if name == 'qlearning' else 'SARSA' if name == 'sarsa' else 'Double Q'
        ax.plot(episodes, smoothed, label=label, color=colors.get(name, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Survival Time (Smoothed)', fontsize=12)
    ax.set_title('Survival Time Learning Curves (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/survival_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/survival_learning_curves.png")

def plot_reward_learning_curves(results):
    """Plot 7: Reward Learning Curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'qlearning': 'green', 'sarsa': 'blue', 'double_q': 'purple'}
    window = 50
    
    for name, data in results.items():
        rewards = data['rewards']
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        episodes = range(window, len(rewards) + 1)
        label = 'Q-Learning' if name == 'qlearning' else 'SARSA' if name == 'sarsa' else 'Double Q'
        ax.plot(episodes, smoothed, label=label, color=colors.get(name, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (Smoothed)', fontsize=12)
    ax.set_title('Reward Learning Curves (ε=0.15 Constant)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/reward_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/reward_learning_curves.png")

def plot_reward_distribution(results):
    """Plot 8: Reward Distribution Box Plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    labels = ['Q-Learning', 'SARSA', 'Double Q-Learning']
    
    reward_data = [results[algo]['rewards'][-500:] for algo in algorithms]
    bp = ax.boxplot(reward_data, labels=labels, patch_artist=True)
    
    box_colors = ['lightgreen', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Total Reward per Episode', fontsize=12)
    ax.set_title('Reward Distribution (Last 500 Episodes)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, algo in enumerate(algorithms):
        mean_reward = np.mean(results[algo]['rewards'][-500:])
        ax.text(i + 1, mean_reward, f'μ={mean_reward:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/reward_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/reward_distribution.png")

def plot_efficiency_curves(results):
    """Plot 9: Reward per Step Efficiency Curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'qlearning': 'green', 'sarsa': 'blue', 'double_q': 'purple'}
    window = 50
    
    for name, data in results.items():
        rewards = data['rewards']
        times = data['times']
        reward_per_step = [r / max(t, 1) for r, t in zip(rewards, times)]
        smoothed = np.convolve(reward_per_step, np.ones(window)/window, mode='valid')
        episodes = range(window, len(reward_per_step) + 1)
        label = 'Q-Learning' if name == 'qlearning' else 'SARSA' if name == 'sarsa' else 'Double Q'
        ax.plot(episodes, smoothed, label=label, color=colors.get(name, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward per Step', fontsize=12)
    ax.set_title('Learning Efficiency (Reward per Step)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/efficiency_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/efficiency_curves.png")

def plot_summary_table(results):
    """Plot 10: Summary Table"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Algorithm', 'Avg Score', 'Best Score', 'Avg Time', 'Best Time', 'Avg Reward', 'Train Time']
    
    data = []
    for algo in ['qlearning', 'sarsa', 'double_q']:
        if algo in results:
            d = results[algo]
            algo_name = 'Q-Learning' if algo == 'qlearning' else 'SARSA' if algo == 'sarsa' else 'Double Q'
            avg_reward = np.mean(d['rewards'][-500:]) if len(d['rewards']) >= 500 else np.mean(d['rewards'])
            data.append([
                algo_name,
                f"{d['avg_score_last_500']:.2f}",
                f"{d['best_score']}",
                f"{np.mean(d['times'][-500:]):.1f}",
                f"{d['best_time']}",
                f"{avg_reward:.1f}",
                f"{d['training_time']:.1f}s"
            ])
    
    data.sort(key=lambda x: float(x[1]), reverse=True)
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i, row in enumerate(data):
        if 'Q-Learning' in row[0]:
            table[(i+1, 0)].set_facecolor('#c8e6c9')
        elif 'SARSA' in row[0]:
            table[(i+1, 0)].set_facecolor('#bbdefb')
        else:
            table[(i+1, 0)].set_facecolor('#f8bbd0')
    
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#333333')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)
    
    ax.set_title('EPSILON-GREEDY ALGORITHMS COMPARISON (ε=0.15 Constant)\n3000 Episodes Training', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('assets/summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: assets/summary_table.png")

def print_detailed_comparison_table(results):
    """Print detailed comparison table in console"""
    
    print("\n" + "="*90)
    print("🏆 EPSILON-GREEDY ALGORITHMS COMPARISON (ε=0.15 Constant) 🏆")
    print("="*90)
    
    print(f"\n{'Algorithm':<20} {'Avg Score':<12} {'Best Score':<12} {'Avg Time':<12} {'Best Time':<12} {'Avg Reward':<12} {'Train Time':<10}")
    print("-"*100)
    
    comparison_data = []
    for algo in ['qlearning', 'sarsa', 'double_q']:
        if algo in results:
            d = results[algo]
            algo_name = 'Q-Learning' if algo == 'qlearning' else 'SARSA' if algo == 'sarsa' else 'Double Q'
            avg_time = np.mean(d['times'][-500:]) if len(d['times']) >= 500 else np.mean(d['times'])
            avg_reward = np.mean(d['rewards'][-500:]) if len(d['rewards']) >= 500 else np.mean(d['rewards'])
            
            comparison_data.append({
                'algo': algo_name,
                'avg_score': d['avg_score_last_500'],
                'best_score': d['best_score'],
                'avg_time': avg_time,
                'best_time': d['best_time'],
                'avg_reward': avg_reward,
                'train_time': d['training_time']
            })
    
    comparison_data.sort(key=lambda x: x['avg_score'], reverse=True)
    
    for i, d in enumerate(comparison_data, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {d['algo']:<17} {d['avg_score']:<12.2f} {d['best_score']:<12} {d['avg_time']:<12.1f} {d['best_time']:<12} {d['avg_reward']:<12.1f} {d['train_time']:<10.1f}s")
    
    print("\n" + "="*90)
    print("📊 REWARD ANALYSIS")
    print("="*90)
    
    for d in comparison_data:
        print(f"\n{d['algo']}:")
        print(f"  • Average Total Reward per Episode: {d['avg_reward']:.1f}")
        print(f"  • Average Score per Episode: {d['avg_score']:.2f}")
        print(f"  • Reward-to-Score Ratio: {(d['avg_reward'] / d['avg_score']):.2f}" if d['avg_score'] > 0 else "  • Reward-to-Score Ratio: N/A")
    
    print("\n" + "="*90)
    print("🏆 BEST IN EACH CATEGORY")
    print("="*90)
    
    best_score = max(comparison_data, key=lambda x: x['avg_score'])
    best_peak = max(comparison_data, key=lambda x: x['best_score'])
    fastest_train = min(comparison_data, key=lambda x: x['train_time'])
    best_survival = max(comparison_data, key=lambda x: x['avg_time'])
    best_reward = max(comparison_data, key=lambda x: x['avg_reward'])
    
    print(f"\n  🏆 Best Average Score:     {best_score['algo']} ({best_score['avg_score']:.2f})")
    print(f"  🎯 Best Peak Score:        {best_peak['algo']} ({best_peak['best_score']})")
    print(f"  ⏱️  Best Survival Time:     {best_survival['algo']} ({best_survival['avg_time']:.1f} steps)")
    print(f"  💰 Best Average Reward:    {best_reward['algo']} ({best_reward['avg_reward']:.1f})")
    print(f"  ⚡ Fastest Training:       {fastest_train['algo']} ({fastest_train['train_time']:.1f}s)")
    
    print("\n" + "="*90)
    print("💡 RECOMMENDATION")
    print("="*90)
    
    winner = comparison_data[0]
    print(f"\n  ✅ BEST ALGORITHM: {winner['algo']}")
    print(f"     • Average Score: {winner['avg_score']:.2f}")
    print(f"     • Average Reward: {winner['avg_reward']:.1f}")
    print(f"     • Training Time: {winner['train_time']:.1f} seconds")
    print(f"     • Best Score: {winner['best_score']}")

def main():
    results = load_results()
    if results is None:
        return
    
    print("\n📊 Generating comparison graphs...")
    print("="*50)
    
    # Generate all plots individually
    plot_score_comparison(results)
    plot_survival_comparison(results)
    plot_training_time_comparison(results)
    plot_reward_comparison(results)
    plot_score_learning_curves(results)
    plot_survival_learning_curves(results)
    plot_reward_learning_curves(results)
    plot_reward_distribution(results)
    plot_efficiency_curves(results)
    plot_summary_table(results)
    
    # Print console comparison
    print("\n" + "="*50)
    print_detailed_comparison_table(results)
    
    print("\n" + "="*50)
    print("✅ All 10 visualizations saved in 'assets/' folder:")
    print("="*50)
    print("  📊 assets/score_comparison.png")
    print("  📊 assets/survival_comparison.png")
    print("  📊 assets/training_time_comparison.png")
    print("  📊 assets/reward_comparison.png")
    print("  📊 assets/score_learning_curves.png")
    print("  📊 assets/survival_learning_curves.png")
    print("  📊 assets/reward_learning_curves.png")
    print("  📊 assets/reward_distribution.png")
    print("  📊 assets/efficiency_curves.png")
    print("  📊 assets/summary_table.png")
    print("="*50)

if __name__ == "__main__":
    main()