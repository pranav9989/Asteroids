# play.py (Updated for 3 Epsilon-Greedy Algorithms)
import pygame
import pickle
import numpy as np
from env import AsteroidsEnv
import os

pygame.init()

CELL = 60
WIDTH = 10
HEIGHT = 10

screen = pygame.display.set_mode((WIDTH*CELL, HEIGHT*CELL))
pygame.display.set_caption("Asteroids RL - Epsilon-Greedy Agents")
clock = pygame.time.Clock()

# Colors
YELLOW = (255, 255, 100)
CYAN = (100, 255, 255)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (100, 255, 100)
BLUE = (100, 100, 255)
PURPLE = (255, 100, 255)
ORANGE = (255, 165, 0)
DARK_GREEN = (0, 150, 0)
DARK_BLUE = (0, 0, 150)

# Agent colors (only epsilon agents)
AGENT_COLORS = {
    'qlearning': GREEN,
    'sarsa': BLUE,
    'double_q': PURPLE
}

font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 36)
small_font = pygame.font.SysFont("Arial", 18)
tiny_font = pygame.font.SysFont("Arial", 14)

def list_available_agents():
    """List all trained agents"""
    agents = ['qlearning', 'sarsa', 'double_q']
    
    available = []
    print("\n" + "="*60)
    print("🎮 AVAILABLE AGENTS (Epsilon-Greedy, ε=0.1 constant)")
    print("="*60)
    
    for i, agent in enumerate(agents, 1):
        filename = f"q_table_{agent}.pkl"
        if os.path.exists(filename):
            available.append(agent)
            algo = "Q-Learning" if agent == "qlearning" else "SARSA" if agent == "sarsa" else "Double Q"
            print(f"  {i}. {algo} ✅")
        else:
            print(f"  {i}. {agent} ❌ (not trained yet)")
    
    return available

def load_q_table(agent_name):
    """Load Q-table for selected agent"""
    filename = f"q_table_{agent_name}.pkl"
    try:
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
        print(f"\n✅ Loaded {agent_name.upper()} Q-table")
        print(f"   Q-table size: {len(q_table)} states")
        return q_table
    except FileNotFoundError:
        print(f"❌ No model found for {agent_name}")
        return {}

def choose_action(state, q_table, num_asteroids):
    """Choose action based on Q-table"""
    if state in q_table:
        q_values = q_table[state]
        if num_asteroids == 0:
            if isinstance(q_values, (list, np.ndarray)):
                q_values = q_values.copy()
                q_values[2] = -float('inf')
        return np.argmax(q_values)
    else:
        # Unknown state - explore with bias
        if num_asteroids > 0:
            return np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.5])
        return np.random.randint(2)

def draw_stars(screen):
    """Draw background stars"""
    for _ in range(50):
        x = np.random.randint(0, WIDTH*CELL)
        y = np.random.randint(0, HEIGHT*CELL)
        pygame.draw.circle(screen, (100, 100, 150), (x, y), 1)

def main():
    available = list_available_agents()
    
    if not available:
        print("\n❌ No trained agents found! Please run train.py first.")
        pygame.quit()
        return
    
    print("\n" + "="*60)
    print("Select agent to play:")
    print("="*60)
    for i, agent in enumerate(available, 1):
        algo = "Q-Learning" if agent == "qlearning" else "SARSA" if agent == "sarsa" else "Double Q"
        print(f"  {i}. {algo}")
    
    while True:
        try:
            choice = int(input("\nEnter choice: ")) - 1
            if 0 <= choice < len(available):
                agent_name = available[choice]
                break
        except ValueError:
            pass
        print("Invalid choice, try again")
    
    q_table = load_q_table(agent_name)
    
    if not q_table:
        print("❌ Failed to load Q-table. Exiting...")
        pygame.quit()
        return
    
    env = AsteroidsEnv(width=WIDTH, height=HEIGHT)
    state = env.reset()
    
    running = True
    game_over = False
    final_score = 0
    final_time = 0
    angle = 0
    shot_timer = 0
    FPS = 10
    
    # Track stats
    episodes_played = 0
    total_score = 0
    total_time = 0
    scores_history = []
    
    print(f"\n🎮 Playing with {agent_name.upper()} Agent")
    print("-"*50)
    print("Controls:")
    print("  • Watch AI play automatically")
    print("  • SPACE or R - Restart after game over")
    print("  • ESC - Quit game")
    print("-"*50)
    print("🤖 Agent uses Epsilon-Greedy (ε=0.1 constant)")
    print("-"*50)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if (event.key == pygame.K_SPACE or event.key == pygame.K_r) and game_over:
                    state = env.reset()
                    game_over = False
                    episodes_played += 1
        
        if not game_over:
            num_asteroids = len(env.asteroids)
            action = choose_action(state, q_table, num_asteroids)
            state, reward, done = env.step(action)
            
            if action == 2:
                shot_timer = 5
            
            if done:
                game_over = True
                final_score = env.score
                final_time = env.time_alive
                total_score += final_score
                total_time += final_time
                scores_history.append(final_score)
                
                avg_score = total_score / episodes_played if episodes_played > 0 else 0
                print(f"Episode {episodes_played + 1}: Score={final_score}, Time={final_time}, Avg={avg_score:.1f}")
        
        # Draw everything
        screen.fill((0, 0, 20))
        draw_stars(screen)
        
        # Draw grid
        for x in range(WIDTH):
            pygame.draw.line(screen, (30, 30, 50), (x*CELL, 0), (x*CELL, HEIGHT*CELL), 1)
        for y in range(HEIGHT):
            pygame.draw.line(screen, (30, 30, 50), (0, y*CELL), (WIDTH*CELL, y*CELL), 1)
        
        # Draw ship
        ship_x = env.ship_x * CELL + CELL//2
        ship_y = (HEIGHT-1)*CELL + CELL//2
        
        color = AGENT_COLORS.get(agent_name, GREEN)
        
        # Draw ship with glow effect
        pygame.draw.polygon(screen, color,
            [(ship_x, ship_y-20), (ship_x-15, ship_y+15), (ship_x+15, ship_y+15)])
        pygame.draw.polygon(screen, (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50)),
            [(ship_x, ship_y-18), (ship_x-12, ship_y+12), (ship_x+12, ship_y+12)], 2)
        
        # Shooting effect
        if shot_timer > 0:
            pygame.draw.line(screen, (255, 255, 0), (ship_x, ship_y - 15), (ship_x, ship_y - 50), 4)
            pygame.draw.line(screen, (255, 100, 0), (ship_x - 2, ship_y - 15), (ship_x - 2, ship_y - 50), 2)
            pygame.draw.line(screen, (255, 100, 0), (ship_x + 2, ship_y - 15), (ship_x + 2, ship_y - 50), 2)
            shot_timer -= 1
        
        # Draw asteroids
        for asteroid in env.asteroids:
            ax = asteroid[0] * CELL + CELL//2
            ay = asteroid[1] * CELL + CELL//2
            
            if asteroid[1] >= HEIGHT - 3:
                color_ast = (200, 80, 80)
                pygame.draw.circle(screen, (255, 0, 0, 50), (ax, ay), 22, 3)
            else:
                color_ast = (120, 120, 140)
            
            pygame.draw.circle(screen, color_ast, (ax, ay), 18)
            pygame.draw.circle(screen, (60, 60, 80), (ax, ay), 18, 2)
            pygame.draw.circle(screen, (80, 80, 100), (ax-5, ay-4), 4)
            pygame.draw.circle(screen, (80, 80, 100), (ax+6, ay+3), 3)
        
        angle = (angle + 3) % 360
        
        # Draw UI Panel
        stats_bg = pygame.Surface((250, 160))
        stats_bg.set_alpha(200)
        stats_bg.fill((0, 0, 0))
        screen.blit(stats_bg, (5, 5))
        
        # Agent info
        algo_display = "Q-Learning" if agent_name == "qlearning" else "SARSA" if agent_name == "sarsa" else "Double Q"
        
        algo_text = font.render(f"{algo_display}", True, color)
        epsilon_text = small_font.render(f"ε-greedy (ε=0.1)", True, color)
        
        # Current game stats
        score_text = font.render(f"Score: {env.score}", True, YELLOW)
        time_text = small_font.render(f"Time: {env.time_alive}", True, CYAN)
        asteroid_text = small_font.render(f"Asteroids: {len(env.asteroids)}", True, WHITE)
        
        # Episode stats
        if episodes_played > 0:
            avg_score = total_score / episodes_played
            avg_time = total_time / episodes_played
            avg_text = small_font.render(f"Avg Score: {avg_score:.1f}", True, GREEN)
            avg_time_text = tiny_font.render(f"Avg Time: {avg_time:.0f}", True, CYAN)
            best_text = tiny_font.render(f"Best: {max(scores_history) if scores_history else 0}", True, YELLOW)
            
            screen.blit(avg_text, (15, 130))
            screen.blit(avg_time_text, (15, 147))
            screen.blit(best_text, (15, 164))
        
        screen.blit(algo_text, (15, 15))
        screen.blit(epsilon_text, (15, 45))
        screen.blit(score_text, (15, 75))
        screen.blit(time_text, (15, 105))
        screen.blit(asteroid_text, (15, 125))
        
        # Game over screen
        if game_over:
            overlay = pygame.Surface((WIDTH*CELL, HEIGHT*CELL))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            game_over_text = big_font.render("GAME OVER", True, RED)
            final_score_text = font.render(f"Final Score: {final_score}", True, YELLOW)
            final_time_text = small_font.render(f"Time: {final_time}", True, CYAN)
            
            if final_score >= 50:
                rating = "🏆 EXCELLENT! 🏆"
                rating_color = (255, 215, 0)
            elif final_score >= 30:
                rating = "👍 GOOD JOB! 👍"
                rating_color = GREEN
            elif final_score >= 15:
                rating = "👌 NOT BAD 👌"
                rating_color = YELLOW
            else:
                rating = "💪 KEEP LEARNING! 💪"
                rating_color = RED
            
            rating_text = small_font.render(rating, True, rating_color)
            restart_text = small_font.render("Press SPACE or R to restart", True, WHITE)
            quit_text = small_font.render("Press ESC to quit", True, WHITE)
            
            center_x = WIDTH*CELL//2
            screen.blit(game_over_text, (center_x - game_over_text.get_width()//2, HEIGHT*CELL//2 - 80))
            screen.blit(final_score_text, (center_x - final_score_text.get_width()//2, HEIGHT*CELL//2 - 30))
            screen.blit(final_time_text, (center_x - final_time_text.get_width()//2, HEIGHT*CELL//2))
            screen.blit(rating_text, (center_x - rating_text.get_width()//2, HEIGHT*CELL//2 + 40))
            screen.blit(restart_text, (center_x - restart_text.get_width()//2, HEIGHT*CELL//2 + 80))
            screen.blit(quit_text, (center_x - quit_text.get_width()//2, HEIGHT*CELL//2 + 110))
        
        pygame.display.update()
        clock.tick(FPS)
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    if episodes_played > 0:
        print(f"Episodes played: {episodes_played}")
        print(f"Average Score: {total_score/episodes_played:.2f}")
        print(f"Best Score: {max(scores_history) if scores_history else 0}")
        print(f"Average Time: {total_time/episodes_played:.1f}")
    
    pygame.quit()
    print("\n👋 Game closed!")

if __name__ == "__main__":
    main()