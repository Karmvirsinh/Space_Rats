
import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions for Localization -----

def get_surroundings(ship, pos):
    """
    Returns a tuple representing the states (0=blocked, 1=open) of the 8 neighboring cells
    (in the order: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right).
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
    Returns the new position after moving from pos in the specified direction.
    """
    deltas = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Returns the opposite move (for backtracking).
    """
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

# ----- Helper Function for Phase 2 -----

def argmax_candidate(prob_dict):
    """
    Returns the candidate (cell) with the highest probability.
    """
    return max(prob_dict, key=prob_dict.get)

# ----- Unified Bot for Stationary Rat -----

def unified_bot_stationary(ship, alpha=0.15, max_steps_phase2=1000, replan_interval=3):
    """
    Runs a unified bot that first localizes itself and then tracks a stationary rat.
    
    Phase 1: Localization using the 8-neighbor sensor reading to filter candidates.
             When a unique candidate remains, backtracks to compute the initial spawn.
    
    Phase 2: Rat Tracking (stationary rat)
             - Initializes a uniform belief distribution over all open cells (except the bot’s spawn).
             - In each iteration, uses the sensor reading (via ping_detector) to update beliefs
               using a Bayesian update.
             - Chooses the candidate with the highest probability as the target.
             - Plans a BFS path toward the target and moves one step along the path.
             - Repeats until the sensor definitively indicates the rat is present
               (i.e. when bot position equals rat position).
    
    Returns (moves, senses, pings, estimated_spawn, true_rat_pos).
    """
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --- Phase 1: Localization ---
    candidate_set = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(candidate_set))  # hidden true spawn for simulation
    bot_pos = true_bot_pos
    move_history = []
    sensor = get_surroundings(ship, bot_pos)
    senses += 1
    candidate_set = {pos for pos in candidate_set if get_surroundings(ship, pos) == sensor}
    print("Phase 1: Localization")
    print("True bot spawn:", true_bot_pos)
    print("Initial sensor:", sensor, "Candidate set size:", len(candidate_set))
    step_local = 0
    max_steps_local = 100
    while len(candidate_set) > 1 and step_local < max_steps_local:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            print("No valid moves for localization; stopping.")
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
        print("Estimated spawn (after backtracking):", estimated_spawn)
    else:
        estimated_spawn = bot_pos
        print("Localization not unique; using current position as spawn:", bot_pos)
    # Bot is now localized.
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Stationary) ---
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_candidates))
    print("Phase 2: Rat Tracking (Stationary)")
    print("True rat spawn:", true_rat_pos)
    # Initialize uniform belief over rat candidates.
    rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}

    steps_since_replan = 0
    current_path = []  # BFS path (list of moves)
    step_phase2 = 0
    while bot_pos != true_rat_pos and moves < max_steps_phase2:
        # Sensor reading update.
        sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist = manhattan_distance(bot_pos, true_rat_pos)
        pings += 1
        print(f"Bot at {bot_pos} | Sensor ping: {sensor_ping} | Distance to rat: {dist}")
        # If sensor indicates and distance is zero, the rat is found.
        if sensor_ping and dist == 0:
            print("Rat captured at", bot_pos)
            break
        # Bayesian update of belief distribution.
        new_probs = {}
        total_prob = 0.0
        for pos, prob in rat_probs.items():
            d = manhattan_distance(bot_pos, pos)
            if d == 0:
                likelihood = 1.0
            else:
                base = np.exp(-alpha * (d - 1))
                likelihood = base if sensor_ping else (1.0 - base)
            new_p = prob * likelihood
            new_probs[pos] = new_p
            total_prob += new_p
        if total_prob > 0:
            for pos in new_probs:
                new_probs[pos] /= total_prob
        rat_probs = new_probs

        # Choose target as the candidate with maximum probability.
        target_candidate = argmax_candidate(rat_probs)
        print(f"Target candidate: {target_candidate} with probability {rat_probs[target_candidate]:.4f}")

        # Replan path if needed.
        if steps_since_replan == 0 or not current_path:
            current_path = bfs_path(ship, bot_pos, target_candidate)
            steps_since_replan = replan_interval
            if not current_path:
                valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
                if valid_moves:
                    chosen_move = random.choice(valid_moves)
                    print("No path found; taking a random valid move:", chosen_move)
                else:
                    print("No valid moves available; terminating Phase 2.")
                    break
            else:
                chosen_move = current_path.pop(0)
        else:
            chosen_move = current_path.pop(0) if current_path else None
            if not chosen_move:
                current_path = bfs_path(ship, bot_pos, target_candidate)
                if not current_path:
                    valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
                    chosen_move = random.choice(valid_moves) if valid_moves else None
                else:
                    chosen_move = current_path.pop(0)
        if not chosen_move:
            print("No move chosen; terminating Phase 2.")
            break
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        print(f"Moving {chosen_move} -> New bot pos: {bot_pos}")
        steps_since_replan = max(steps_since_replan - 1, 0)
        step_phase2 += 1

    print(f"Phase 2 complete: Bot at {bot_pos}, True rat at {true_rat_pos}")
    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, est_spawn, true_rat = unified_bot_stationary(ship, alpha=0.15, max_steps_phase2=1000)
    print("Final stats: Moves:", moves, "Senses:", senses, "Pings:", pings)
