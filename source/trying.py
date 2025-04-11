
import numpy as np
import random
import heapq
import pygame
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

##########################################
# Helper Functions: Localization (Phase 1)
##########################################

def get_surroundings(ship, pos):
    """
    Returns a tuple representing the states (0=blocked, 1=open) of the 8 neighboring cells.
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
    Updates candidate positions in bot_knowledge after a move.
    If the bot moved, retains only candidates that would also move; otherwise, those that remain.
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

##########################################
# Helper Functions: Visualization with Pygame
##########################################

def visualize_state_pygame(ship, bot_pos, rat_pos, rat_probs, screen, cell_size, font):
    """
    Visualizes the current state using Pygame:
      - Draws the ship grid (open: white; blocked: dark gray).
      - Colors each open cell based on its probability: 0 = white, 1 = red.
      - Draws the bot (red circle) and the rat (green circle).
      - Displays a key at the bottom.
    """
    FPS=60
    D = ship.shape[0]
    screen.fill((0, 0, 0))
    for r in range(D):
        for c in range(D):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            if ship[r, c] == 1:
                if (r, c) in rat_probs:
                    prob = rat_probs[(r, c)]
                    # Interpolate: 0 -> white, 1 -> red.
                    color = (255, int((1 - prob) * 255), int((1 - prob) * 255))
                else:
                    color = (255, 255, 255)
            else:
                color = (50, 50, 50)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200,200,200), rect, 1)
    # Draw bot and rat.
    bot_center = (int(bot_pos[1]*cell_size + cell_size/2), int(bot_pos[0]*cell_size + cell_size/2))
    pygame.draw.circle(screen, (255, 0, 0), bot_center, cell_size//2)
    rat_center = (int(rat_pos[1]*cell_size + cell_size/2), int(rat_pos[0]*cell_size + cell_size/2))
    pygame.draw.circle(screen, (0, 255, 0), rat_center, cell_size//2)
    # Draw key.
    key_text = font.render("Key: Cell color intensity = probability rat is here", True, (255,255,255))
    screen.blit(key_text, (5, D * cell_size + 5))
    pygame.display.flip()
    clock = pygame.time.Clock()
    # clock.tick(FPS)

##########################################
# Helper Functions: BFS-based Simulation Rollouts
##########################################

def weighted_sample(rat_probs, num_samples):
    """
    Returns a list of samples drawn from rat_probs using weighted sampling.
    """
    candidates = list(rat_probs.keys())
    weights = [rat_probs[c] for c in candidates]
    return random.choices(candidates, weights=weights, k=num_samples)

def simulate_rollout_cost(target, rat_probs, ship, num_samples=10):
    """
    Estimates expected cost-to-go from target by sampling rat positions (based on rat_probs)
    and averaging the BFS distances from target to each sample.
    """
    samples = weighted_sample(rat_probs, num_samples)
    total = 0
    count = 0
    for sample in samples:
        path = bfs_path(ship, target, sample)
        if path:
            total += len(path)
            count += 1
        else:
            total += 1000
            count += 1
    return total / count if count > 0 else float('inf')

def candidate_total_cost(ship, bot_pos, candidate, rat_probs):
    """
    Computes total cost for a candidate target as:
       cost = (BFS distance from bot_pos to candidate) + 1 + (expected future cost from candidate via rollouts)
    """
    path = bfs_path(ship, bot_pos, candidate)
    if not path:
        return float('inf')
    travel_cost = len(path)
    rollout_cost = simulate_rollout_cost(candidate, rat_probs, ship, num_samples=10)
    return travel_cost + 1 + rollout_cost

def argmin_cost_candidate(ship, bot_pos, rat_probs):
    """
    Returns the candidate from rat_probs with the minimum total cost.
    """
    best_candidate = None
    best_cost = float('inf')
    for candidate in rat_probs:
        cost = candidate_total_cost(ship, bot_pos, candidate, rat_probs)
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return best_candidate

##########################################
# Enhanced Custom Bot with Pygame Visualization and Modified Bayesian Update
##########################################

def custom_bot_enhanced(ship, alpha=0.15, gamma=0.5, max_steps_phase2=1000, replan_interval=5, visualize=True):
    """
    Enhanced custom bot that:
      1. Localizes itself using 8-neighbor sensor filtering.
      2. Tracks a stationary rat using a modified Bayesian update where, if no ping is received,
         the likelihood is computed as (1 - base)^gamma (with gamma < 1 to reduce penalty),
         ensuring that the expected future cost is lower.
      3. Always heads toward the cell with the highest probability.
      4. Uses BFS for path planning.
      5. Uses receding horizon planning (replan_interval) and oscillation detection.
      6. Visualizes the grid with Pygame: cells are colored from white (low probability) to red (high probability)
         and a key shows that cell color intensity represents the probability.
    
    Returns (moves, senses, pings, estimated_spawn, true_rat_pos).
    """
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --------------------
    # Phase 1: Localization
    # --------------------
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

    # --------------------
    # Phase 2: Enhanced Rat Tracking (Stationary)
    # --------------------
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_candidates))
    print("Phase 2: Rat Tracking (Enhanced)")
    print("True rat spawn:", true_rat_pos)
    rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}
    
    # For accumulating sensor evidence (not used in this version but could be extended).
    ping_count = 1
    
    # Set up Pygame.
    cell_size = 20
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((D * cell_size, D * cell_size + 40))
        pygame.display.set_caption("Enhanced Bot Visualization")
        font = pygame.font.SysFont("Arial", 14)
    
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
        
        sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
        d = manhattan_distance(bot_pos, true_rat_pos)
        pings += 1
        print(f"Bot at {bot_pos} | Sensor ping: {sensor_ping} | Distance to rat: {d}")
        if sensor_ping and d == 0:
            print("Rat captured at", bot_pos)
            break
        
        # Bayesian update of belief:
        # First, zero out the probability for the cell the bot is on.
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
                if sensor_ping:
                    likelihood = base ** ping_count
                else:
                    # When no ping, use (1 - base)^gamma (gamma < 1 reduces the penalty)
                    likelihood = (1.0 - base) ** gamma
            new_p = prob * likelihood
            new_probs[pos] = new_p
            total_prob += new_p
        if total_prob > 0:
            for pos in new_probs:
                new_probs[pos] /= total_prob
        rat_probs = new_probs

        # Select target: choose the candidate with the highest probability.
        current_target = max(rat_probs, key=rat_probs.get)
        print(f"Selected target: {current_target} with probability {rat_probs[current_target]:.2f}")
        
        # Replan path using BFS.
        if steps_since_replan == 0 or not current_path:
            current_path = bfs_path(ship, bot_pos, current_target)
            steps_since_replan = replan_interval
            if not current_path:
                valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
                if valid_moves:
                    chosen_move = random.choice(valid_moves)
                    print("No BFS path found; taking a random move:", chosen_move)
                else:
                    print("No valid moves available; terminating Phase 2.")
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
        
        # if visualize:
            # visualize_state_pygame(ship, bot_pos, true_rat_pos, rat_probs, screen, cell_size, font)
    
    # if visualize:
    #     pygame.time.wait(2000)
    #     pygame.quit()
    
    print("Phase 2 complete: Bot at", bot_pos, "True rat at", true_rat_pos)
    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, est_spawn, true_rat = custom_bot_enhanced(ship, alpha=0.15, gamma=0.5, max_steps_phase2=1000, replan_interval=5, visualize=True)
    print(f"Final stats: Moves: {moves}, Senses: {senses}, Pings: {pings}")
