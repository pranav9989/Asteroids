import pygame
import pickle
import numpy as np
from env import AsteroidsEnv

pygame.init()

CELL = 60
WIDTH = 10
HEIGHT = 10

screen = pygame.display.set_mode((WIDTH*CELL, HEIGHT*CELL))
pygame.display.set_caption("Asteroids RL - Score & Time Tracker")
clock = pygame.time.Clock()

# Initialize font for displaying stats
font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 36)

env = AsteroidsEnv()

# load Q table
try:
    with open("q_table.pkl","rb") as f:
        q = pickle.load(f)
    print("Q-table loaded successfully!")
except FileNotFoundError:
    print("No Q-table found. Using random actions.")
    q = {}

def choose_action(state):
    if state in q:
        return np.argmax(q[state])
    else:
        return np.random.randint(3)  # Random action if no Q-table

state = env.reset()

running = True
game_over = False
final_score = 0
final_time = 0

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_over:
                # Reset game on spacebar after game over
                state = env.reset()
                game_over = False

    if not game_over:
        action = choose_action(state)
        state, reward, done = env.step(action)
        
        if done:
            game_over = True
            final_score = env.score
            final_time = env.time_alive

    screen.fill((0,0,20))

    # draw ship
    ship_x = env.ship_x * CELL + CELL//2
    ship_y = (HEIGHT-1)*CELL + CELL//2

    pygame.draw.polygon(screen,(0,255,0),
        [(ship_x,ship_y-20),(ship_x-20,ship_y+20),(ship_x+20,ship_y+20)])

    # draw asteroids
    for asteroid in env.asteroids:

        ax = asteroid[0] * CELL + CELL//2
        ay = asteroid[1] * CELL + CELL//2

        pygame.draw.circle(screen,(200,200,200),(ax,ay),20)

    # Draw score and time
    score_text = font.render(f"Score: {env.score}", True, (255, 255, 100))
    time_text = font.render(f"Time: {env.time_alive}", True, (100, 255, 255))
    
    screen.blit(score_text, (10, 10))
    screen.blit(time_text, (10, 40))

    # Draw game over screen if done
    if game_over:
        # Create semi-transparent overlay
        overlay = pygame.Surface((WIDTH*CELL, HEIGHT*CELL))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = big_font.render("GAME OVER", True, (255, 0, 0))
        final_score_text = font.render(f"Final Score: {final_score}", True, (255, 255, 255))
        final_time_text = font.render(f"Time Survived: {final_time}", True, (255, 255, 255))
        restart_text = font.render("Press SPACE to restart", True, (200, 200, 200))
        
        # Center text
        screen.blit(game_over_text, (WIDTH*CELL//2 - game_over_text.get_width()//2, HEIGHT*CELL//2 - 60))
        screen.blit(final_score_text, (WIDTH*CELL//2 - final_score_text.get_width()//2, HEIGHT*CELL//2 - 20))
        screen.blit(final_time_text, (WIDTH*CELL//2 - final_time_text.get_width()//2, HEIGHT*CELL//2 + 10))
        screen.blit(restart_text, (WIDTH*CELL//2 - restart_text.get_width()//2, HEIGHT*CELL//2 + 50))

    pygame.display.update()
    clock.tick(5)  # 5 FPS for visibility

pygame.quit()