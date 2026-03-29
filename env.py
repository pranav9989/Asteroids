import random

class AsteroidsEnv:

    def __init__(self, width=10, height=10, max_asteroids=3):

        self.width = width
        self.height = height
        self.max_asteroids = max_asteroids

        self.reset()

    def reset(self):

        self.ship_x = self.width // 2
        self.asteroids = []
        
        # Track metrics
        self.score = 0
        self.time_alive = 0

        self.done = False

        return self.get_state()

    def spawn_asteroid(self):

        if len(self.asteroids) < self.max_asteroids:

            x = random.randint(0, self.width-1)

            self.asteroids.append([x,0])

    def get_state(self):

        # only track nearest asteroid for Q-table simplicity
        if len(self.asteroids) == 0:
            return (self.ship_x, -1, -1)

        nearest = min(self.asteroids, key=lambda a: a[1])

        return (self.ship_x, nearest[0], nearest[1])

    def step(self, action):

        reward = 0

        # move ship
        if action == 0:
            self.ship_x = max(0, self.ship_x - 1)

        elif action == 1:
            self.ship_x = min(self.width-1, self.ship_x + 1)

        elif action == 2:  # shoot

            for asteroid in self.asteroids:
                if asteroid[0] == self.ship_x:
                    reward += 10
                    self.score += 1  # Increment score when asteroid destroyed
                    self.asteroids.remove(asteroid)
                    break

        # spawn new asteroid randomly
        if random.random() < 0.3:
            self.spawn_asteroid()

        # move asteroids
        for asteroid in self.asteroids:
            asteroid[1] += 1

        # collision check
        for asteroid in self.asteroids:

            if asteroid[1] == self.height-1 and asteroid[0] == self.ship_x:
                reward -= 100
                self.done = True

        # remove off-screen asteroids
        self.asteroids = [a for a in self.asteroids if a[1] < self.height]

        # Track survival time
        self.time_alive += 1
        reward += 1

        return self.get_state(), reward, self.done