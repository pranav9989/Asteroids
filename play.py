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
pygame.display.set_caption("Asteroids RL - Destroy Asteroids!")
clock = pygame.time.Clock()

# Initialize fonts
font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 36)
small_font = pygame.font.SysFont("Arial", 18)

# Colors
YELLOW = (255, 255, 100)
CYAN = (100, 255, 255)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (100, 255, 100)
ORANGE = (255, 165, 0)

# Load images
try:
    spaceship_img = pygame.image.load("assets/spaceship.jpeg")
    asteroid_img = pygame.image.load("assets/asteroid.jpeg")
    
    # Scale images to fit cell size
    spaceship_img = pygame.transform.scale(spaceship_img, (CELL - 10, CELL - 10))
    asteroid_img = pygame.transform.scale(asteroid_img, (CELL - 10, CELL - 10))
    
    print("✅ Images loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading images: {e}")
    print("Make sure you have 'assets/spaceship.jpeg' and 'assets/asteroid.jpeg'")
    spaceship_img = None
    asteroid_img = None

env = AsteroidsEnv(width=WIDTH, height=HEIGHT)

# Load Q table
try:
    with open("q_table.pkl", "rb") as f:
        q = pickle.load(f)
    print("✅ Q-table loaded successfully!")
    print("🎯 Agent is trained to DESTROY ASTEROIDS!")
except FileNotFoundError:
    print("❌ No Q-table found. Using random actions.")
    q = {}

def choose_action(state):
    """Choose action based on trained Q-table"""
    if state in q:
        return np.argmax(q[state])
    else:
        # Smart random fallback for unseen states
        if len(state) >= 3:
            num_asteroids = state[2] if state[2] != -1 else 0
            if num_asteroids > 0:
                return np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])  # Prefer shooting
        return np.random.randint(3)

state = env.reset()

running = True
game_over = False
final_score = 0
final_time = 0

# For asteroid animation
angle = 0

# FPS control
FPS = 10

# For shooting animation
shot_timer = 0

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if (event.key == pygame.K_SPACE or event.key == pygame.K_r) and game_over:
                # Reset game on spacebar or R after game over
                state = env.reset()
                game_over = False

    if not game_over:
        action = choose_action(state)
        state, reward, done = env.step(action)
        
        # Add shooting effect
        if action == 2:
            shot_timer = 5
        
        if done:
            game_over = True
            final_score = env.score
            final_time = env.time_alive

    screen.fill((0, 0, 20))

    # Draw grid lines (helps visualize positions)
    for x in range(WIDTH):
        pygame.draw.line(screen, (40, 40, 60), (x*CELL, 0), (x*CELL, HEIGHT*CELL), 1)
    for y in range(HEIGHT):
        pygame.draw.line(screen, (40, 40, 60), (0, y*CELL), (WIDTH*CELL, y*CELL), 1)

    # Draw ship
    ship_x = env.ship_x * CELL + CELL//2
    ship_y = (HEIGHT-1)*CELL + CELL//2
    
    if spaceship_img:
        ship_rect = spaceship_img.get_rect(center=(ship_x, ship_y))
        screen.blit(spaceship_img, ship_rect)
        
        # Shooting effect (laser)
        if shot_timer > 0:
            pygame.draw.line(screen, (255, 255, 0), (ship_x, ship_y - 15), (ship_x, ship_y - 60), 3)
            pygame.draw.line(screen, (255, 100, 0), (ship_x - 2, ship_y - 15), (ship_x - 2, ship_y - 60), 2)
            pygame.draw.line(screen, (255, 100, 0), (ship_x + 2, ship_y - 15), (ship_x + 2, ship_y - 60), 2)
            shot_timer -= 1
    else:
        # Fallback to triangle
        pygame.draw.polygon(screen, (0, 255, 0),
            [(ship_x, ship_y-20), (ship_x-20, ship_y+20), (ship_x+20, ship_y+20)])

    # Draw asteroids
    for asteroid in env.asteroids:
        ax = asteroid[0] * CELL + CELL//2
        ay = asteroid[1] * CELL + CELL//2
        
        if asteroid_img:
            # Rotate asteroids for animation
            rotated_asteroid = pygame.transform.rotate(asteroid_img, angle)
            asteroid_rect = rotated_asteroid.get_rect(center=(ax, ay))
            screen.blit(rotated_asteroid, asteroid_rect)
            
            # Red glow for asteroids near bottom
            if asteroid[1] >= HEIGHT - 3:
                pygame.draw.circle(screen, (255, 0, 0, 50), (ax, ay), 25, 2)
        else:
            # Fallback to circle
            color = (200, 200, 200)
            if asteroid[1] >= HEIGHT - 3:
                color = (255, 100, 100)
            pygame.draw.circle(screen, color, (ax, ay), 20)
    
    # Update rotation angle
    angle = (angle + 3) % 360

    # Draw statistics panel
    stats_bg = pygame.Surface((200, 100))
    stats_bg.set_alpha(180)
    stats_bg.fill((0, 0, 0))
    screen.blit(stats_bg, (5, 5))
    
    # Draw stats
    score_text = font.render(f"🎯 Score: {env.score}", True, YELLOW)
    time_text = small_font.render(f"⏱️ Time: {env.time_alive}", True, CYAN)
    asteroid_count = small_font.render(f"☄ Asteroids: {len(env.asteroids)}", True, WHITE)
    
    screen.blit(score_text, (15, 15))
    screen.blit(time_text, (15, 50))
    screen.blit(asteroid_count, (15, 80))

    # Hint text
    if not game_over and len(env.asteroids) > 0:
        hint_text = small_font.render("🔫 SHOOT! (+10 pts)", True, GREEN)
        screen.blit(hint_text, (WIDTH*CELL - hint_text.get_width() - 10, 10))

    # Game over screen
    if game_over:
        overlay = pygame.Surface((WIDTH*CELL, HEIGHT*CELL))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        game_over_text = big_font.render("GAME OVER", True, RED)
        final_score_text = font.render(f"🎯 Final Score: {final_score}", True, YELLOW)
        final_time_text = small_font.render(f"⏱️ Time Survived: {final_time}", True, CYAN)
        
        # Performance rating
        if final_score >= 30:
            rating = "🏆 EXCELLENT! 🏆"
            rating_color = (255, 215, 0)
        elif final_score >= 15:
            rating = "👍 GOOD JOB! 👍"
            rating_color = GREEN
        elif final_score >= 5:
            rating = "👌 NOT BAD 👌"
            rating_color = YELLOW
        else:
            rating = "💪 KEEP TRAINING! 💪"
            rating_color = RED
        
        rating_text = small_font.render(rating, True, rating_color)
        restart_text = small_font.render("Press SPACE or R to restart", True, WHITE)
        
        # Center all text
        screen.blit(game_over_text, (WIDTH*CELL//2 - game_over_text.get_width()//2, HEIGHT*CELL//2 - 80))
        screen.blit(final_score_text, (WIDTH*CELL//2 - final_score_text.get_width()//2, HEIGHT*CELL//2 - 30))
        screen.blit(final_time_text, (WIDTH*CELL//2 - final_time_text.get_width()//2, HEIGHT*CELL//2))
        screen.blit(rating_text, (WIDTH*CELL//2 - rating_text.get_width()//2, HEIGHT*CELL//2 + 40))
        screen.blit(restart_text, (WIDTH*CELL//2 - restart_text.get_width()//2, HEIGHT*CELL//2 + 80))

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
print("👋 Game closed!")