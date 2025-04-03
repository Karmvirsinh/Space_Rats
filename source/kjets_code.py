import random
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


# -------------------------------
# Ship Generation and Utilities
# -------------------------------

def generate_ship(D):
    """
    Generates a D x D maze (ship) with outer edges blocked.
    Uses a randomized growing process.
    """
    grid = [[0 for _ in range(D)] for _ in range(D)]
    # Choose a random starting cell (not on border)
    start_i = random.randint(1, D - 2)
    start_j = random.randint(1, D - 2)
    grid[start_i][start_j] = 1

    while True:
        candidates = []
        for i in range(1, D - 1):
            for j in range(1, D - 1):
                if grid[i][j] == 0:
                    open_neighbors = 0
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if grid[ni][nj] == 1:
                            open_neighbors += 1
                    if open_neighbors == 1:
                        candidates.append((i, j))
        if not candidates:
            break
        cell = random.choice(candidates)
        grid[cell[0]][cell[1]] = 1

    # Optionally open up some dead ends to improve connectivity.
    dead_ends = []
    for i in range(D):
        for j in range(D):
            if grid[i][j] == 1:
                count = 0
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
                        count += 1
                if count == 1:
                    dead_ends.append((i, j))
    for cell in dead_ends:
        if random.random() < 0.5:
            i, j = cell
            neighbors = []
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 0:
                    neighbors.append((ni, nj))
            if neighbors:
                new_cell = random.choice(neighbors)
                grid[new_cell[0]][new_cell[1]] = 1
    return grid


def get_neighbors(pos, grid):
    """Return the cardinal (up, down, left, right) neighbors that are open."""
    D = len(grid)
    i, j = pos
    nbrs = []
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
            nbrs.append((ni, nj))
    return nbrs


def manhattan_distance(a, b):
    """Compute the Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -------------------------------
# Sensor Functions
# -------------------------------

def sense_blocked_neighbors(pos, grid):
    """
    Senses the number of blocked cells among the eight surrounding cells.
    A cell is blocked if it is outside the grid or if grid value is 0.
    """
    D = len(grid)
    i, j = pos
    count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= D or nj < 0 or nj >= D or grid[ni][nj] == 0:
                count += 1
    return count


def space_rat_ping(bot_pos, rat_pos, alpha):
    """
    Returns whether a ping is heard from the space rat detector.
    If the bot and rat share the same cell, it always pings.
    Otherwise, the probability is e^(-α*(d-1)) where d is Manhattan distance.
    """
    if bot_pos == rat_pos:
        return True
    d = manhattan_distance(bot_pos, rat_pos)
    prob = math.exp(-alpha * (d - 1))
    return random.random() < prob


# -------------------------------
# Baseline Bot for Space Rats
# -------------------------------

class BaselineSpaceRatsBot:
    """
    Baseline bot for Project 2.

    Phase 1 (Localization):
      - Maintains a knowledge base (kb_bot) of possible positions (all open inner cells).
      - Alternates between sensing (to get number of blocked neighbors) and moving.
      - When sensing, it prunes kb_bot to only those cells with the same sensed value.
      - When moving, it chooses the cardinal direction that is most common among its candidate positions.
      - After the move, kb_bot is pruned based on whether the move would be blocked or not.
      - When kb_bot reduces to a single candidate, localization is complete.

    Phase 2 (Space Rat Tracking):
      - The space rat knowledge base (kb_rat) is maintained as a probability distribution over open cells.
      - Each timestep, the bot uses the space rat detector. Using Bayes’ rule:
            If ping received: likelihood L(x) = e^(-α*(d(bot,x)-1))
            If no ping:    likelihood L(x) = 1 - e^(-α*(d(bot,x)-1))
        then update and renormalize.
      - The bot moves one step toward the cell with highest probability.

    If moving_rat is True, after each bot action the space rat moves randomly.
    """

    def __init__(self, grid, alpha, moving_rat=False):
        self.grid = grid
        self.D = len(grid)
        self.alpha = alpha
        self.moving_rat = moving_rat

        # List of open inner cells
        self.open_cells = [(i, j) for i in range(1, self.D - 1) for j in range(1, self.D - 1) if grid[i][j] == 1]
        # Initialize bot’s true position randomly
        self.true_pos = random.choice(self.open_cells)
        # Initialize space rat’s true position (different from bot)
        rat_candidates = [c for c in self.open_cells if c != self.true_pos]
        self.rat_pos = random.choice(rat_candidates)

        # Action counters
        self.move_count = 0
        self.blocked_sense_count = 0
        self.rat_detector_count = 0

        # Phase 1: Bot localization KB (all open inner cells)
        self.kb_bot = set(self.open_cells)
        # For alternating sensing and moving in Phase 1
        self.phase1_sense_next = True

        # Phase 2: Space rat tracking KB (uniform distribution)
        self.kb_rat = {cell: 1 / len(self.open_cells) for cell in self.open_cells}

        # Start in Phase 1
        self.phase = 1

    # ----- Phase 1: Localization -----
    def update_bot_localization(self, sensor_val):
        """Prune kb_bot based on sensed blocked neighbors value."""
        new_kb = {cell for cell in self.kb_bot if sense_blocked_neighbors(cell, self.grid) == sensor_val}
        self.kb_bot = new_kb

    def choose_movement_direction(self):
        """
        For each cardinal direction, count candidate cells that have an open neighbor in that direction.
        Return the direction with the highest count.
        """
        directions = {(-1, 0): 0, (1, 0): 0, (0, 1): 0, (0, -1): 0}
        for cell in self.kb_bot:
            i, j = cell
            for d in directions:
                di, dj = d
                ni, nj = i + di, j + dj
                if 0 <= ni < self.D and 0 <= nj < self.D and self.grid[ni][nj] == 1:
                    directions[d] += 1
        best_dir = max(directions, key=directions.get)
        return best_dir

    def attempt_move(self, direction):
        """
        Attempt to move the bot in a cardinal direction.
        Return True if successful, False otherwise.
        """
        i, j = self.true_pos
        di, dj = direction
        new_pos = (i + di, j + dj)
        if 0 <= new_pos[0] < self.D and 0 <= new_pos[1] < self.D and self.grid[new_pos[0]][new_pos[1]] == 1:
            self.true_pos = new_pos
            return True
        return False

    def update_kb_after_move(self, direction, success):
        """
        If move succeeded, remove from kb_bot any candidate where that move would be blocked.
        If move failed, remove those where the move would be open.
        """
        new_kb = set()
        for cell in self.kb_bot:
            i, j = cell
            di, dj = direction
            ni, nj = i + di, j + dj
            can_move = (0 <= ni < self.D and 0 <= nj < self.D and self.grid[ni][nj] == 1)
            if (success and can_move) or (not success and not can_move):
                new_kb.add(cell)
        self.kb_bot = new_kb

    # ----- Phase 2: Rat Tracking -----
    def update_rat_kb(self, sensor_ping):
        """
        Update the space rat KB using Bayes’ rule.
        For each cell, the likelihood is:
          - L(x)= e^(-α*(d(bot,x)-1)) if ping is received,
          - L(x)= 1 - e^(-α*(d(bot,x)-1)) otherwise.
        """
        new_probs = {}
        total = 0
        for cell, prior in self.kb_rat.items():
            d = manhattan_distance(self.true_pos, cell)
            # If bot and cell coincide, sensor is definitive.
            if self.true_pos == cell:
                likelihood = 1.0 if sensor_ping else 0.0
            else:
                p_ping = math.exp(-self.alpha * (d - 1))
                likelihood = p_ping if sensor_ping else (1 - p_ping)
            new_probs[cell] = prior * likelihood
            total += new_probs[cell]
        if total > 0:
            for cell in new_probs:
                new_probs[cell] /= total
        else:
            # In rare cases, reset to uniform
            uniform = 1 / len(self.kb_rat)
            new_probs = {cell: uniform for cell in self.kb_rat}
        self.kb_rat = new_probs

    def choose_rat_target(self):
        """Return the cell with the highest probability in kb_rat."""
        return max(self.kb_rat, key=self.kb_rat.get)

    def move_towards(self, target):
        """
        Move one step toward the target cell using a greedy Manhattan distance move.
        In this baseline bot, if two moves are possible, choose randomly among those that reduce distance.
        """
        i, j = self.true_pos
        ti, tj = target
        candidates = []
        if ti < i:
            candidates.append((-1, 0))
        elif ti > i:
            candidates.append((1, 0))
        if tj < j:
            candidates.append((0, -1))
        elif tj > j:
            candidates.append((0, 1))
        if candidates:
            move = random.choice(candidates)
            self.attempt_move(move)
            self.move_count += 1

    def rat_move_randomly(self):
        """For the moving rat variant, move the rat to a random open neighbor."""
        nbrs = get_neighbors(self.rat_pos, self.grid)
        if nbrs:
            self.rat_pos = random.choice(nbrs)

    # -------------------------------
    # Simulation Step & Visualization
    # -------------------------------

    def step(self):
        """
        Execute one timestep:
          - In Phase 1, alternate between sensing and moving until localization is complete.
          - In Phase 2, use the space rat detector, update kb_rat, and move toward the target.
          - If the bot catches the rat (positions coincide), simulation ends.
        Returns True if simulation should continue, False if complete.
        """
        if self.phase == 1:
            if self.phase1_sense_next:
                sensor_val = sense_blocked_neighbors(self.true_pos, self.grid)
                self.blocked_sense_count += 1
                self.update_bot_localization(sensor_val)
                # If only one candidate remains, localization is complete.
                if len(self.kb_bot) == 1:
                    self.phase = 2
                self.phase1_sense_next = False
            else:
                direction = self.choose_movement_direction()
                success = self.attempt_move(direction)
                self.move_count += 1
                self.update_kb_after_move(direction, success)
                # If candidate set reduces to one, complete localization.
                if len(self.kb_bot) == 1:
                    self.phase = 2
                self.phase1_sense_next = True
        elif self.phase == 2:
            # Use the rat detector.
            self.rat_detector_count += 1
            ping = space_rat_ping(self.true_pos, self.rat_pos, self.alpha)
            # If bot catches the rat, end simulation.
            if self.true_pos == self.rat_pos:
                return False
            self.update_rat_kb(ping)
            target = self.choose_rat_target()
            self.move_towards(target)
            if self.true_pos == self.rat_pos:
                return False
            # For moving rat variant, update rat's position.
            if self.moving_rat:
                self.rat_move_randomly()
        return True

    def get_state(self):
        """
        Returns current simulation state as a dictionary for visualization:
          - grid, bot position, rat position,
          - current phase, and the bot's localization KB (if in Phase 1)
          - and the rat probability distribution as a 2D array (if in Phase 2)
        """
        state = {
            "grid": self.grid,
            "bot": self.true_pos,
            "rat": self.rat_pos,
            "phase": self.phase,
            "kb_bot": self.kb_bot if self.phase == 1 else None,
            "kb_rat": None
        }
        if self.phase == 2:
            # Create a 2D probability array for the rat distribution.
            prob = np.zeros((self.D, self.D))
            for (i, j), p in self.kb_rat.items():
                prob[i, j] = p
            state["kb_rat"] = prob
        return state

    def run(self, visualize=False, delay=0.2):
        """
        Run the simulation until the bot catches the rat.
        If visualize is True, update a matplotlib plot in real time.
        Returns a tuple: (move_count, blocked_sense_count, rat_detector_count)
        """
        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 6))
        continue_sim = True
        while continue_sim:
            if visualize:
                state = self.get_state()
                ax.clear()
                grid = np.array(state["grid"])
                ax.imshow(grid, cmap='gray_r')
                # Plot bot position (blue) and rat position (red)
                bot_y, bot_x = state["bot"]
                rat_y, rat_x = state["rat"]
                ax.plot(bot_x, bot_y, 'o', markersize=12, color='blue', label="Bot")
                ax.plot(rat_x, rat_y, 'o', markersize=12, color='red', label="Rat")
                # In Phase 1, plot the candidate KB for localization as green dots.
                if state["phase"] == 1 and state["kb_bot"]:
                    xs = [j for (i, j) in state["kb_bot"]]
                    ys = [i for (i, j) in state["kb_bot"]]
                    ax.plot(xs, ys, 'o', markersize=4, color='green', label="KB Candidates")
                # In Phase 2, overlay the rat probability distribution.
                if state["phase"] == 2 and state["kb_rat"] is not None:
                    # Overlay as a transparent heatmap.
                    ax.imshow(state["kb_rat"], cmap='hot', alpha=0.5)
                ax.set_title(
                    f"Phase {state['phase']}\nMoves: {self.move_count}, Senses: {self.blocked_sense_count}, Detector: {self.rat_detector_count}")
                ax.legend(loc='upper right')
                plt.pause(delay)
            continue_sim = self.step()
        if visualize:
            plt.ioff()
            plt.show()
        return self.move_count, self.blocked_sense_count, self.rat_detector_count


# -------------------------------
# Improved Bot Variant
# -------------------------------

class ImprovedSpaceRatsBot(BaselineSpaceRatsBot):
    """
    An improved bot that in Phase 2 chooses the neighbor that minimizes the Manhattan
    distance to the target (instead of random tie-breaking).
    """

    def move_towards(self, target):
        i, j = self.true_pos
        nbrs = get_neighbors(self.true_pos, self.grid)
        if not nbrs:
            return
        best = None
        best_dist = float('inf')
        for nbr in nbrs:
            d = manhattan_distance(nbr, target)
            if d < best_dist:
                best = nbr
                best_dist = d
        if best is not None:
            self.true_pos = best
            self.move_count += 1


# -------------------------------
# Simulation and Analysis Routine
# -------------------------------

def simulate_project2(bot_class, alpha=0.1, moving_rat=False, trials=10, visualize=False):
    """
    Runs multiple simulation trials and collects average action counts.
    If visualize is True, the first trial is shown in real time.
    Returns lists of movement, sensing, and detector action counts.
    """
    moves_list = []
    sense_list = []
    detector_list = []
    D = 30
    for trial in range(trials):
        grid = generate_ship(D)
        bot = bot_class(grid, alpha, moving_rat=moving_rat)
        # Visualize only the first trial
        if visualize and trial == 0:
            print("Visualizing trial 1...")
            moves, senses, detectors = bot.run(visualize=True, delay=0.3)
        else:
            moves, senses, detectors = bot.run(visualize=False)
        moves_list.append(moves)
        sense_list.append(senses)
        detector_list.append(detectors)
    return moves_list, sense_list, detector_list


def main():
    print("Project 2: Space Rats Simulation")
    print("Choose simulation mode:")
    print("1: Static Space Rat with Baseline Bot")
    print("2: Moving Space Rat with Baseline Bot")
    print("3: Moving Space Rat with Improved Bot")
    mode = input("Enter 1, 2, or 3: ").strip()
    alpha = float(input("Enter sensor sensitivity α (e.g., 0.1): ").strip())
    trials = int(input("Enter number of simulation trials (e.g., 10): ").strip())
    viz_choice = input("Visualize the first trial? (y/n): ").strip().lower()
    visualize = (viz_choice == "y")

    if mode == "1":
        bot_class = BaselineSpaceRatsBot
        moving = False
        print("Running simulation: Static Space Rat with Baseline Bot")
    elif mode == "2":
        bot_class = BaselineSpaceRatsBot
        moving = True
        print("Running simulation: Moving Space Rat with Baseline Bot")
    elif mode == "3":
        bot_class = ImprovedSpaceRatsBot
        moving = True
        print("Running simulation: Moving Space Rat with Improved Bot")
    else:
        print("Invalid mode selected. Exiting.")
        return

    moves, senses, detectors = simulate_project2(bot_class, alpha=alpha, moving_rat=moving, trials=trials,
                                                 visualize=visualize)
    print("\nSimulation Results (averaged over trials):")
    print("Average movement actions:", sum(moves) / len(moves))
    print("Average blocked-cell sensing actions:", sum(senses) / len(senses))
    print("Average space rat detector actions:", sum(detectors) / len(detectors))

    # Plot histograms for analysis.
    plt.figure()
    plt.hist(moves, bins=10, edgecolor='black')
    plt.xlabel("Movement Actions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Movement Actions")
    plt.show()

    plt.figure()
    plt.hist(senses, bins=10, edgecolor='black')
    plt.xlabel("Blocked Neighbor Sensing Actions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sensing Actions")
    plt.show()

    plt.figure()
    plt.hist(detectors, bins=10, edgecolor='black')
    plt.xlabel("Space Rat Detector Actions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Detector Actions")
    plt.show()

    # -------------------------------
    # Analysis Discussion (for writeup)
    # -------------------------------
    print("\n=== Analysis Discussion ===")
    print("""
1) Space Rat Knowledge Base Update:
   - When a ping is received, the likelihood for cell x is L(x) = e^(-α*(d(bot,x)-1)).
   - When no ping is received, L(x) = 1 - e^(-α*(d(bot,x)-1)).
   - The posterior probability p(x) ∝ prior(x) * L(x) is then renormalized.
   This update is derived from Bayes’ rule and accounts for the sensor model.

2) Bot Design and Algorithm:
   - Phase 1: The bot maintains a KB of its possible positions. It alternates between sensing (pruning candidates based on the number of blocked neighbors) and moving (choosing the most likely direction). This guarantees that over time the bot converges to its actual position.
   - Phase 2: The bot uses a uniform initial distribution over all open cells as its rat KB. With each space rat detector reading, it updates the distribution via Bayes’ rule and moves one step toward the most likely location.
   - In the improved variant, the bot deterministically picks the move that minimizes the Manhattan distance to the target.

3) Performance Evaluation:
   - By running simulations over varying values of α (e.g., between 0 and 0.2), one can compare the average number of moves, sensing, and detector actions.
   - As α increases, the sensor becomes less informative because the probability of a ping decreases, leading to a potential increase in required actions. When α is too small, the sensor pings almost always, which also reduces the information gained.
   - The provided histograms and average counts can be plotted as a function of α to compare performance between the baseline and improved bots.

4) Moving Space Rat Updates:
   - When the rat moves, its KB update must convolve the previous distribution with the rat’s motion model (here assumed to be a uniform random move to an adjacent open cell).
   - In our simulation we have not performed a full convolution update but instead simply move the rat randomly after each bot action.
   - This change typically increases the difficulty of tracking and catching the rat, which is evident in the increased action counts in simulations.

You can use these outputs and visualizations to draw conclusions regarding the effectiveness of the design choices and sensor parameters.
""")


if __name__ == "__main__":
    main()