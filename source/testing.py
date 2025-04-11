import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions for Localization -----

def get_surroundings(ship, pos):
    """
    Returns a tuple representing the states (0=blocked, 1=open) of the 8 neighboring cells 
    (order: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right).
    Out-of-bound cells are considered blocked.
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
    deltas = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Returns the reverse (opposite) of the given move.
    """
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

# ----- Helper for Phase 2: Cost Calculation -----

def candidate_cost(ship, bot_pos, candidate, rat_probs):
    """
    Computes an approximate cost of using candidate as the target.
    Cost = (shortest-path distance from bot_pos to candidate) 
         + 1 (for a sensing action) 
         + (weighted average Manhattan distance from candidate to all rat candidate cells,
            weighted by rat_probs).
    If no path exists (i.e. bfs_path returns empty), the cost is set to infinity.
    """
    path = bfs_path(ship, bot_pos, candidate)
    if not path:
        return float('inf')
    path_cost = len(path)
    future_cost = sum(rat_probs[r] * manhattan_distance(candidate, r) for r in rat_probs)
    return path_cost + 1 + future_cost

def argmin_cost_candidate(ship, bot_pos, rat_probs):
    """
    Returns the candidate in rat_probs with the minimum cost, according to candidate_cost.
    """
    best_candidate = None
    best_cost = float('inf')
    for candidate in rat_probs:
        cost = candidate_cost(ship, bot_pos, candidate, rat_probs)
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return best_candidate

# ----- Custom Bot Implementation -----

def unified_bot_stationary(ship, alpha=0.15, max_steps_phase2=1000):
    """
    Runs a custom bot that first localizes itself and then tracks a stationary rat using a cost-based heuristic.
    
    Phase 1: Localization
      - All open cells are candidate spawn locations.
      - The bot “senses” its 8 neighbors and filters candidates to those with matching patterns.
      - It then makes random valid moves (updating the candidate set by simulating the move on each candidate)
        until only one candidate remains, then backtracks to compute the estimated spawn.
    
    Phase 2: Rat Tracking (Stationary Rat)
      - Initializes a uniform belief distribution over all open cells (except the bot's spawn) as candidate rat positions.
      - At each iteration, performs a Bayesian update of the belief based on the sensor reading.
      - Computes the cost for each candidate as:
             cost = (shortest-path distance from bot to candidate) + 1 + (weighted average distance from candidate to rat candidates).
      - Chooses the candidate with the lowest cost as the target.
      - Plans a path (via BFS) toward that target and moves one step along that path.
      - Repeats until the sensor definitively indicates the rat is captured (i.e. bot position equals rat position).
    
    Returns a tuple: (moves, senses, pings, estimated_spawn, true_rat_pos)
    """
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --- Phase 1: Localization ---
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
        print(f"Move: {chosen_move}, Bot pos: {bot_pos}, Sensor: {sensor}, Candidates left: {len(candidate_set)}")
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

    # --- Phase 2: Custom Rat Tracking (Stationary Rat) ---
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_candidates))
    print("Phase 2: Rat Tracking (Custom Bot)")
    print("True rat spawn:", true_rat_pos)
    # Initialize uniform belief over rat candidates.
    rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}
    
    step_phase2 = 0
    while bot_pos != true_rat_pos and moves < max_steps_phase2:
        sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
        d = manhattan_distance(bot_pos, true_rat_pos)
        pings += 1
        # Terminate when sensor indicates capture.
        if sensor_ping and d == 0:
            print("Rat captured at", bot_pos)
            break
        
        # --- Bayesian Update of Belief ---
        new_probs = {}
        total_prob = 0.0
        for pos, prob in rat_probs.items():
            d_candidate = manhattan_distance(bot_pos, pos)
            if d_candidate == 0:
                likelihood = 1.0
            else:
                base = np.exp(-alpha * (d_candidate - 1))
                likelihood = base if sensor_ping else (1.0 - base)
            new_p = prob * likelihood
            new_probs[pos] = new_p
            total_prob += new_p
        if total_prob > 0:
            for pos in new_probs:
                new_probs[pos] /= total_prob
        rat_probs = new_probs

        # --- Target Selection Using Cost Function ---
        target_candidate = argmin_cost_candidate(ship, bot_pos, rat_probs)
        target_cost = candidate_cost(ship, bot_pos, target_candidate, rat_probs)
        print(f"Bot at {bot_pos}: Sensor ping: {sensor_ping}, Distance to rat: {d}")
        print(f"Target candidate: {target_candidate} with cost {target_cost:.2f}")

        # --- Path Planning ---
        path = bfs_path(ship, bot_pos, target_candidate)
        if not path:
            valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                chosen_move = random.choice(valid_moves)
                print("No path found; taking a random valid move:", chosen_move)
            else:
                print("No valid moves; terminating Phase 2.")
                break
        else:
            chosen_move = path[0]
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        print(f"Moving {chosen_move} -> New bot pos: {bot_pos}")
        step_phase2 += 1

    print("Phase 2 complete: Bot at", bot_pos, "True rat at", true_rat_pos)
    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, est_spawn, true_rat = unified_bot_stationary(ship, alpha=0.15, max_steps_phase2=1000)
    print(f"Final stats: Moves: {moves}, Senses: {senses}, Pings: {pings}")
