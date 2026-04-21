# play_enhanced.py (Updated for 3 algorithms)
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
pygame.display.set_caption("Asteroids RL - 3 Algorithm Comparison")
clock = pygame.time.Clock()

# Colors
YELLOW = (255, 255, 100)
CYAN = (100, 255, 255)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (100, 255, 100)
BLUE = (100, 100, 255)
PURPLE = (255, 100, 255)

# Algorithm colors
ALGO_COLORS = {
    'qlearning': GREEN,
    'sarsa': BLUE,
    'double_q': PURPLE
}

font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 36)
small_font = pygame.font.SysFont("Arial", 18)

def load_algorithm():
    """Let user choose which algorithm to play"""
    
    algorithms = ['qlearning', 'sarsa', 'double_q']
    available = []
    
    print("\n🎮 Available Q-tables:")
    for algo in algorithms:
        filename = f"q_table_{algo}.pkl"
        if os.path.exists(filename):
            available.append(algo)
            print(f"  ✅ {algo.upper()}")
        else:
            print(f"  ❌ {algo.upper()} (not found - run training first)")
    
    if not available:
        print("\n❌ No Q-tables found! Please run train_all_algorithms.py first.")
        return None
    
    print("\nSelect algorithm to play:")
    print("  1. Q-LEARNING")
    print("  2. SARSA")
    print("  3. DOUBLE Q-LEARNING")
    
    while True:
        try:
            choice = int(input("Enter choice (1-3): ")) - 1
            if 0 <= choice < len(available):
                return available[choice]
        except:
            pass
        print("Invalid choice, try again")

def load_q_table(algo_name):
    """Load Q-table for selected algorithm"""
    filename = f"q_table_{algo_name}.pkl"
    try:
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
        print(f"✅ Loaded {algo_name.upper()} Q-table")
        return q_table
    except FileNotFoundError:
        print(f"❌ No Q-table found for {algo_name}")
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
        if num_asteroids > 0:
            return np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])
        return np.random.randint(2)

def main():
    algo_name = load_algorithm()
    if algo_name is None:
        pygame.quit()
        return
    
    q_table = load_q_table(algo_name)
    
    env = AsteroidsEnv(width=WIDTH, height=HEIGHT)
    state = env.reset()
    
    running = True
    game_over = False
    final_score = 0
    final_time = 0
    angle = 0
    shot_timer = 0
    FPS = 10
    
    print(f"\n🎮 Playing with {algo_name.upper()} agent")
    print("SPACE/R to restart after game over | ESC to quit")
    
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
        
        # Draw everything
        screen.fill((0, 0, 20))
        
        # Draw grid
        for x in range(WIDTH):
            pygame.draw.line(screen, (40, 40, 60), (x*CELL, 0), (x*CELL, HEIGHT*CELL), 1)
        for y in range(HEIGHT):
            pygame.draw.line(screen, (40, 40, 60), (0, y*CELL), (WIDTH*CELL, y*CELL), 1)
        
        # Draw ship
        ship_x = env.ship_x * CELL + CELL//2
        ship_y = (HEIGHT-1)*CELL + CELL//2
        
        # Simple triangle ship (no images needed)
        color = ALGO_COLORS.get(algo_name, GREEN)
        pygame.draw.polygon(screen, color,
            [(ship_x, ship_y-20), (ship_x-15, ship_y+15), (ship_x+15, ship_y+15)])
        
        if shot_timer > 0:
            pygame.draw.line(screen, (255, 255, 0), (ship_x, ship_y - 15), (ship_x, ship_y - 50), 3)
            shot_timer -= 1
        
        # Draw asteroids
        for asteroid in env.asteroids:
            ax = asteroid[0] * CELL + CELL//2
            ay = asteroid[1] * CELL + CELL//2
            
            color = (150, 150, 150)
            if asteroid[1] >= HEIGHT - 3:
                color = (200, 100, 100)
            pygame.draw.circle(screen, color, (ax, ay), 18)
            pygame.draw.circle(screen, (80, 80, 80), (ax, ay), 18, 2)
        
        angle = (angle + 3) % 360
        
        # Draw UI
        stats_bg = pygame.Surface((220, 120))
        stats_bg.set_alpha(180)
        stats_bg.fill((0, 0, 0))
        screen.blit(stats_bg, (5, 5))
        
        algo_text = font.render(f"{algo_name.upper()}", True, ALGO_COLORS.get(algo_name, WHITE))
        score_text = font.render(f"Score: {env.score}", True, YELLOW)
        time_text = small_font.render(f"Time: {env.time_alive}", True, CYAN)
        asteroid_text = small_font.render(f"Asteroids: {len(env.asteroids)}", True, WHITE)
        
        screen.blit(algo_text, (15, 15))
        screen.blit(score_text, (15, 50))
        screen.blit(time_text, (15, 80))
        screen.blit(asteroid_text, (15, 105))
        
        # Game over screen
        if game_over:
            overlay = pygame.Surface((WIDTH*CELL, HEIGHT*CELL))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            game_over_text = big_font.render("GAME OVER", True, RED)
            final_score_text = font.render(f"Final Score: {final_score}", True, YELLOW)
            final_time_text = small_font.render(f"Time: {final_time}", True, CYAN)
            restart_text = small_font.render("Press SPACE or R to restart", True, WHITE)
            quit_text = small_font.render("Press ESC to quit", True, WHITE)
            
            screen.blit(game_over_text, (WIDTH*CELL//2 - game_over_text.get_width()//2, HEIGHT*CELL//2 - 60))
            screen.blit(final_score_text, (WIDTH*CELL//2 - final_score_text.get_width()//2, HEIGHT*CELL//2 - 10))
            screen.blit(final_time_text, (WIDTH*CELL//2 - final_time_text.get_width()//2, HEIGHT*CELL//2 + 20))
            screen.blit(restart_text, (WIDTH*CELL//2 - restart_text.get_width()//2, HEIGHT*CELL//2 + 60))
            screen.blit(quit_text, (WIDTH*CELL//2 - quit_text.get_width()//2, HEIGHT*CELL//2 + 90))
        
        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()
    print("👋 Game closed!")

if __name__ == "__main__":
    main()