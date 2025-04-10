import numpy as np
import random
import pygame
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

#########################################
# Helper Functions: Localization (Phase 1)
#########################################

def get_surroundings(ship, pos):
    """
    Returns a tuple representing the states (0 = blocked, 1 = open) of the 8 neighboring cells.
    Order: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right.
    Out-of-bound positions are treated as blocked.
    """
    D = ship.shape[0]
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    pattern = []
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])
        else:
            pattern.append(0)
    return tuple(pattern)

def add_move(pos, direction):
    """
    Returns the new position after moving from pos in the given direction.
    """
    deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Returns the reverse (opposite) of the given move.
    """
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

def update_knowledge_after_move(ship, bot_knowledge, direction, bot_moved):
    """
    Simulate the move on all candidate positions.
    If the bot moved, keep only candidates that would also move; if not, keep those that remain.
    """
    new_knowledge = set()
    for pos in bot_knowledge:
        newpos = move(ship, pos, direction)
        if bot_moved:
            if newpos != pos:
                new_knowledge.add(newpos)
        else:
            if newpos == pos:
                new_knowledge.add(pos)
    return new_knowledge

#########################################
# Helper Functions: Visualization with Pygame
#########################################

def visualize_state_pygame(ship, bot_pos, rat_pos, rat_probs, screen, cell_size, font):
    """
    Uses Pygame to display:
      - The ship grid (open cells: white; blocked: dark gray).
      - Each cell's probability (from rat_probs) is used to color that cell:
           probability 0 -> white; probability 1 -> pure red.
      - The bot's position is drawn as a red circle; the rat as a green circle.
      - A key is displayed at the bottom.
    """
    D = ship.shape[0]
    FPS=60
    # Fill background.
    screen.fill((0, 0, 0))
    # Draw grid cells.
    for r in range(D):
        for c in range(D):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            if ship[r, c] == 1:
                # If this cell is in rat_probs, compute color based on probability.
                if (r, c) in rat_probs:
                    # Map probability 0 -> white, 1 -> red.
                    prob = rat_probs[(r, c)]
                    # Linearly interpolate: color = (255, (1-prob)*255, (1-prob)*255)
                    color = (255, int((1 - prob) * 255), int((1 - prob) * 255))
                else:
                    color = (255, 255, 255)  # white
            else:
                color = (50, 50, 50)  # blocked cells: dark gray
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # cell border

    # Draw bot (red circle)
    bot_center = (int(bot_pos[1] * cell_size + cell_size/2), int(bot_pos[0] * cell_size + cell_size/2))
    pygame.draw.circle(screen, (255, 0, 0), bot_center, cell_size//2)
    # Draw rat (green circle)
    rat_center = (int(rat_pos[1] * cell_size + cell_size/2), int(rat_pos[0] * cell_size + cell_size/2))
    pygame.draw.circle(screen, (0, 255, 0), rat_center, cell_size//2)
    
    # Draw key at bottom.
    key_text = font.render("Key: Cell color intensity = probability rat is here", True, (255,255,255))
    screen.blit(key_text, (5, D * cell_size + 5))
    
    pygame.display.flip()
    clock = pygame.time.Clock()
    clock.tick(FPS)

#########################################
# Enhanced Custom Bot: Using BFS for Path Planning and Targeting the Highest Probability Cell
#########################################

def custom_bot_enhanced(ship, alpha=0.15, max_steps_phase2=1000, replan_interval=5, visualize=True):
    """
    Enhanced bot that:
      1. Localizes itself using 8-neighbor sensor filtering.
      2. Tracks a stationary rat using:
         - Bayesian update of belief,
         - Zeroing visited cells (setting their probability to zero),
         - Target selection: always choose the cell with the highest probability,
         - BFS search for path planning,
         - Receding horizon planning (replan every replan_interval moves) and oscillation detection.
      3. Visualizes the grid and belief distribution using Pygame (cells colored based on probability).
    
    Returns (moves, senses, pings, estimated_spawn, true_rat_pos).
    """
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    ##################################
    # Phase 1: Localization
    ##################################
    candidate_set = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(candidate_set))
    bot_pos = true_bot_pos
    move_history = []
    sensor = get_surroundings(ship, bot_pos)
    senses += 1
    candidate_set = {pos for pos in candidate_set if get_surroundings(ship, pos) == sensor}
    print("Phase 1: Localization")
    print("True bot spawn:", true_bot_pos)
    print("Initial sensor:", sensor, "Candidates:", len(candidate_set))
    step_local = 0
    max_steps_local = 100
    while len(candidate_set) > 1 and step_local < max_steps_local:
        valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        sensor = get_surroundings(ship, bot_pos)
        senses += 1
        new_candidates = set()
        for pos in candidate_set:
            new_pos = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos) == sensor:
                new_candidates.add(new_pos)
        candidate_set = new_candidates
        print(f"Localization move {chosen_move}, Bot pos: {bot_pos}, Sensor: {sensor}, Candidates: {len(candidate_set)}")
        step_local += 1
    if len(candidate_set) == 1:
        unique_candidate = candidate_set.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print("Estimated spawn (backtracked):", estimated_spawn)
    else:
        estimated_spawn = bot_pos
        print("Localization not unique; using current bot pos:", bot_pos)
    bot_pos = estimated_spawn

    ##################################
    # Phase 2: Rat Tracking (Stationary)
    ##################################
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_candidates))
    print("Phase 2: Rat Tracking (Enhanced)")
    print("True rat spawn:", true_rat_pos)
    # Initialize uniform belief over candidate rat positions.
    rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}
    
    # Set up Pygame visualization.
    cell_size = 20
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((D * cell_size, D * cell_size + 40))
        pygame.display.set_caption("Enhanced Bot Visualization")
        font = pygame.font.SysFont("Arial", 14)
    
    # Receding horizon variables.
    current_target = None
    current_path = []
    steps_since_replan = 0
    recent_positions = []
    
    while bot_pos != true_rat_pos and moves < max_steps_phase2:
        # Process Pygame events.
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return moves, senses, pings, estimated_spawn, true_rat_pos
        
        # Sensor reading update.
        sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
        d = manhattan_distance(bot_pos, true_rat_pos)
        pings += 1
        print(f"Bot at {bot_pos} | Sensor ping: {sensor_ping} | Distance to rat: {d}")
        if sensor_ping and d == 0:
            print("Rat captured at", bot_pos)
            break
        
        # Bayesian update of belief.
        # First, set the probability for the current cell to 0 because the rat isn't there.
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0
        new_probs = {}
        total_prob = 0.0
        for pos, prob in rat_probs.items():
            d_candidate = manhattan_distance(bot_pos, pos)
            if d_candidate == 0:
                likelihood = 1.0
            else:
                base = np.exp(-alpha * (d_candidate - 1))
                # For this simplified version, we don't accumulate evidence—just use the current sensor.
                likelihood = base if sensor_ping else (1.0 - base)
            new_p = prob * likelihood
            new_probs[pos] = new_p
            total_prob += new_p
        if total_prob > 0:
            for pos in new_probs:
                new_probs[pos] /= total_prob
        rat_probs = new_probs
        
        # Target selection: choose the candidate with the highest probability.
        current_target = max(rat_probs, key=rat_probs.get)
        print(f"Selected target (highest probability): {current_target} with probability {rat_probs[current_target]:.2f}")
        
        # Replanning: Use BFS (not A*) to plan a path.
        if steps_since_replan == 0 or not current_path:
            current_path = bfs_path(ship, bot_pos, current_target)
            steps_since_replan = replan_interval
            if not current_path:
                valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
                if valid_moves:
                    chosen_move = random.choice(valid_moves)
                    print("No BFS path found; taking a random move:", chosen_move)
                else:
                    print("No valid moves; terminating Phase 2.")
                    break
            else:
                chosen_move = current_path.pop(0)
        else:
            chosen_move = current_path.pop(0) if current_path else None
            if not chosen_move:
                current_path = bfs_path(ship, bot_pos, current_target)
                chosen_move = current_path.pop(0) if current_path else None
        if not chosen_move:
            print("No move chosen; terminating Phase 2.")
            break
        
        new_bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        # Oscillation detection.
        if new_bot_pos in recent_positions:
            print("Oscillation detected at", new_bot_pos, "; forcing replan.")
            current_path = []
            steps_since_replan = 0
        bot_pos = new_bot_pos
        recent_positions.append(bot_pos)
        if len(recent_positions) > 5:
            recent_positions.pop(0)
        print(f"Moving {chosen_move} -> New bot pos: {bot_pos}")
        steps_since_replan = max(steps_since_replan - 1, 0)
        
        if visualize:
            visualize_state_pygame(ship, bot_pos, true_rat_pos, rat_probs, screen, cell_size, font)
    
    if visualize:
        pygame.time.wait(2000)
        pygame.quit()
    
    print("Phase 2 complete: Bot at", bot_pos, "True rat at", true_rat_pos)
    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, est_spawn, true_rat = custom_bot_enhanced(ship, alpha=0.15, max_steps_phase2=1000, replan_interval=5, visualize=True)
    print(f"Final stats: Moves: {moves}, Senses: {senses}, Pings: {pings}")
