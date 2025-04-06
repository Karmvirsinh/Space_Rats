import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions for Phase 1 (Localization) -----

def get_surroundings(ship, pos):
    """
    Returns a tuple representing the 8-neighbor pattern (0=blocked, 1=open) around pos.
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
    deltas = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Returns the reverse of a given move.
    """
    rev = {'up':'down', 'down':'up', 'left':'right', 'right':'left'}
    return rev[direction]

# ----- Helper Functions for Phase 2 (Rat Tracking) -----

def weighted_centroid(prob_dict):
    """
    Computes the weighted centroid of a belief distribution.
    """
    total = sum(prob_dict.values())
    if total == 0:
        return None
    r_sum = sum(pos[0] * prob for pos, prob in prob_dict.items())
    c_sum = sum(pos[1] * prob for pos, prob in prob_dict.items())
    return (int(round(r_sum / total)), int(round(c_sum / total)))

def argmax_candidate(prob_dict):
    """
    Returns the candidate (cell) with the highest probability.
    """
    return max(prob_dict, key=prob_dict.get)

# ----- Unified Bot for Stationary Rat -----

def unified_bot(ship, alpha=0.15, rat_moving=False, replan_interval=3, max_steps_phase2=1000):
    """
    Runs a unified bot that first localizes itself and then tracks a stationary rat.
    (Set rat_moving=False to simulate a stationary rat.)
    Returns (moves, senses, pings, estimated_spawn, true_rat_pos)
    """
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --- Phase 1: Localization ---
    candidate_set = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    # For simulation: choose true bot spawn (hidden to bot)
    true_bot_pos = random.choice(list(candidate_set))
    move_history = []
    current_sensor = get_surroundings(ship, true_bot_pos)
    senses += 1
    candidate_set = {pos for pos in candidate_set if get_surroundings(ship, pos) == current_sensor}
    print(f"Phase 1: True bot spawn: {true_bot_pos}")
    print(f"Initial sensor: {current_sensor}, candidate set size: {len(candidate_set)}")
    
    while len(candidate_set) > 1:
        valid_moves = [m for m in ['up','down','left','right'] if move(ship, true_bot_pos, m) != true_bot_pos]
        if not valid_moves:
            print("No valid moves available for localization.")
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        true_bot_pos = move(ship, true_bot_pos, chosen_move)
        moves += 1
        new_sensor = get_surroundings(ship, true_bot_pos)
        senses += 1
        print(f"Moved {chosen_move} to {true_bot_pos}, sensor: {new_sensor}")
        new_candidate_set = set()
        for pos in candidate_set:
            new_pos = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos) == new_sensor:
                new_candidate_set.add(new_pos)
        candidate_set = new_candidate_set
        print(f"Candidate set size: {len(candidate_set)}")
    
    if len(candidate_set) == 1:
        unique_candidate = candidate_set.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print(f"Estimated spawn: {estimated_spawn}")
    else:
        estimated_spawn = true_bot_pos
        print("Localization did not converge uniquely; using current position.")
    
    # Now the bot is localized.
    bot_pos = estimated_spawn

    # --- Phase 2: Efficient Rat Tracking (Stationary Rat) ---
    # All open cells except bot's spawn are possible rat locations.
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_candidates))
    print(f"Phase 2: True rat spawn: {true_rat_pos}")
    # Initialize uniform belief distribution over rat candidates.
    rat_probs = {pos: 1.0/len(rat_candidates) for pos in rat_candidates}

    steps_since_replan = 0
    current_path = []  # current planned BFS path
    # Loop until the sensor definitively indicates the rat is at the bot's cell.
    while moves < max_steps_phase2:
        # For a stationary rat, skip prediction.
        # Sensor reading update.
        sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
        current_dist = manhattan_distance(bot_pos, true_rat_pos)
        pings += 1
        print(f"Bot at {bot_pos} | Sensor ping: {sensor_ping} | Distance to rat: {current_dist}")
        if sensor_ping and current_dist == 0:
            print(f"Rat captured at {bot_pos}!")
            break
        
        # --- Bayesian Update ---
        new_rat_probs = {}
        total_prob = 0.0
        for pos, prob in rat_probs.items():
            d = manhattan_distance(bot_pos, pos)
            if d == 0:
                likelihood = 1.0
            else:
                base = np.exp(-alpha * (d - 1))
                likelihood = base if sensor_ping else (1.0 - base)
            new_p = prob * likelihood
            new_rat_probs[pos] = new_p
            total_prob += new_p
        if total_prob > 0:
            for pos in new_rat_probs:
                new_rat_probs[pos] /= total_prob
        rat_probs = new_rat_probs

        # Choose target as candidate with maximum probability.
        target_candidate = argmax_candidate(rat_probs)
        print(f"Target candidate: {target_candidate} with probability {rat_probs[target_candidate]:.4f}")

        # Replan path every replan_interval moves or if no current path.
        if steps_since_replan == 0 or not current_path:
            current_path = bfs_path(ship, bot_pos, target_candidate)
            steps_since_replan = replan_interval
            if not current_path:
                valid_moves = [m for m in ['up','down','left','right'] if move(ship, bot_pos, m) != bot_pos]
                if valid_moves:
                    chosen_move = random.choice(valid_moves)
                    print("No path to target; taking a random valid move.")
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
        new_bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        print(f"Moving {chosen_move} from {bot_pos} to {new_bot_pos} toward target {target_candidate}")
        bot_pos = new_bot_pos
        steps_since_replan = max(steps_since_replan - 1, 0)
    
    print(f"Phase 2 complete: Bot at {bot_pos}, True rat at {true_rat_pos}")
    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    # For stationary rat, set rat_moving=False.
    moves, senses, pings, estimated_spawn, true_rat_pos = unified_bot(ship, alpha=0.15, rat_moving=False)
    print(f"Final stats: Moves: {moves}, Senses: {senses}, Pings: {pings}")
