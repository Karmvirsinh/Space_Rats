import numpy as np
import random
import math


def generate_ship(size):
    """
    Generates a spaceship grid layout with blocked edges and navigable paths.
    :param size: Size of the grid (e.g., 30 for a 30x30 grid).
    :return: The generated grid as a NumPy array (0 = blocked, 1 = open).
    """
    grid = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        grid[0][i] = grid[size - 1][i] = grid[i][0] = grid[i][size - 1] = 0

    start_x, start_y = random.randint(1, size - 2), random.randint(1, size - 2)
    grid[start_x][start_y] = 1

    def get_neighbors(x, y):
        directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for nx, ny in directions if 1 <= nx < size - 1 and 1 <= ny < size - 1]

    frontier = [(start_x, start_y)]
    while frontier:
        current = random.choice(frontier)
        x, y = current
        neighbors = get_neighbors(x, y)
        open_neighbors = [(nx, ny) for nx, ny in neighbors if grid[nx][ny] == 1]
        if len(open_neighbors) < 2:
            grid[x][y] = 1
            for nx, ny in neighbors:
                if grid[nx][ny] == 0:
                    frontier.append((nx, ny))
        frontier.remove(current)

    return np.array(grid)


class BaselineBot:
    def __init__(self, ship_grid, alpha=0.1):
        self.grid = ship_grid
        self.size = len(ship_grid)
        self.alpha = alpha
        self.knowledge_base = {(x, y) for x in range(1, self.size - 1) for y in range(1, self.size - 1) if
                               self.grid[x][y] == 1}
        self.bot_position = random.choice(list(self.knowledge_base))
        self.rat_position = random.choice(list(self.knowledge_base))
        self.initial_bot_position = self.bot_position
        self.initial_rat_position = self.rat_position

    def sense_blocked_neighbors(self, pos):
        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return sum(1 for nx, ny in neighbors if self.grid[nx][ny] == 0)

    def update_localization(self):
        blocked_count = self.sense_blocked_neighbors(self.bot_position)
        new_kb = {pos for pos in self.knowledge_base if self.sense_blocked_neighbors(pos) == blocked_count}

        if len(new_kb) < len(self.knowledge_base):
            self.knowledge_base = new_kb
            print(f"[Localization] Possible locations left: {len(self.knowledge_base)}")
        else:
            print("[Localization] No progress made, moving randomly.")
            self.force_move()

        if len(self.knowledge_base) == 1:
            self.bot_position = list(self.knowledge_base)[0]
            print(f"[Localization] Bot identified its location as {self.bot_position}")

    def force_move(self):
        open_dirs = [(dx, dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if
                     self.grid[self.bot_position[0] + dx][self.bot_position[1] + dy] == 1]
        if open_dirs:
            move = random.choice(open_dirs)
            self.bot_position = (self.bot_position[0] + move[0], self.bot_position[1] + move[1])

    def rat_sensor_ping(self):
        distance = abs(self.bot_position[0] - self.rat_position[0]) + abs(self.bot_position[1] - self.rat_position[1])
        probability = math.exp(-self.alpha * (distance - 1))
        return random.random() < probability

    def track_rat(self):
        print(f"[Tracking] Bot at {self.bot_position}, searching for rat.")
        rat_likelihood = {pos: self.rat_sensor_ping() for pos in self.knowledge_base}

        if not rat_likelihood:
            print("[Tracking] No valid locations to search for rat! Moving randomly.")
            self.force_move()
        else:
            self.rat_position = max(rat_likelihood, key=rat_likelihood.get)
            self.bot_position = self.rat_position

        print(f"[Tracking] Moving to {self.bot_position}, possible rat location.")

    def run(self):
        steps = 0
        while self.bot_position != self.rat_position:
            if len(self.knowledge_base) > 1:
                self.update_localization()
            else:
                self.track_rat()
            steps += 1
            if steps > 1000:
                print("[Error] Infinite loop detected. Terminating.")
                break

        print(f"Initial Bot Location: {self.initial_bot_position}")
        print(f"Initial Rat Location: {self.initial_rat_position}")
        print(f"Bot Identified Location: {self.bot_position}")
        print(f"Rat Caught at: {self.rat_position} in {steps} steps!")

        return steps


if __name__ == "__main__":
    ship = generate_ship(30)
    bot = BaselineBot(ship)
    bot.run()
