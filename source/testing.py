import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# --- Helper functions for surroundings-based localization ---

def get_surroundings(ship, pos):
    """
    Returns the state (0 = blocked, 1 = open) of the 8 neighbors around pos.
    The order is: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right.
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
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

# --- Baseline Bot Implementation ---

def baseline_bot(ship, alpha=0.15):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # -----------------------
    # Phase 1: Localization
    # -----------------------
    # Candidate set: all open cells on the ship.
    candidate_set = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    
    # For simulation purposes, choose the true spawn randomly.
    true_bot_pos = random.choice(list(candidate_set))
    
    # Record move history (the moves taken after spawn).
    move_history = []
    
    # Get initial sensor reading (the 8-neighbor pattern) at the true spawn.
    current_sensor = get_surroundings(ship, true_bot_pos)
    senses += 1
    # Filter candidate set to only those cells whose surroundings match.
    candidate_set = { pos for pos in candidate_set if get_surroundings(ship, pos) == current_sensor }
    print(f"Phase 1: True Bot Spawn: {true_bot_pos}")
    print(f"Initial sensor reading: {current_sensor}, Candidate set size: {len(candidate_set)}")
    
    # Now, repeatedly move the bot and update the candidate set.
    while len(candidate_set) > 1:
        # Determine valid moves from the current true position (those that change the position).
        valid_moves = [m for m in ['up','down','left','right'] if move(ship, true_bot_pos, m) != true_bot_pos]
        if not valid_moves:
            print("No valid moves available; stopping localization.")
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        
        # Update the true bot position.
        true_bot_pos = move(ship, true_bot_pos, chosen_move)
        moves += 1
        
        # Get new sensor reading at the new true position.
        new_sensor = get_surroundings(ship, true_bot_pos)
        senses += 1
        print(f"Moved {chosen_move} to {true_bot_pos}, new sensor: {new_sensor}")
        
        # Update candidate set by simulating the move on every candidate.
        new_candidate_set = set()
        for pos in candidate_set:
            new_pos = move(ship, pos, chosen_move)
            # Keep candidate if its new surroundings match the new sensor reading.
            if get_surroundings(ship, new_pos) == new_sensor:
                new_candidate_set.add(new_pos)
        candidate_set = new_candidate_set
        print(f"Candidate set size after move: {len(candidate_set)}")
    
    # Once only one candidate remains, backtrack to deduce the initial spawn.
    if len(candidate_set) == 1:
        unique_candidate = candidate_set.pop()
        print(f"Unique candidate for current position: {unique_candidate}")
        # Backtrack through the move history.
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print(f"Estimated initial spawn location: {estimated_spawn}")
    else:
        print("Localization did not converge to a unique candidate; using current position as estimate.")
        estimated_spawn = true_bot_pos
    
    print(f"Phase 1 done. Estimated spawn: {estimated_spawn}")
    
    # -------------------------------
    # Phase 2: Rat Tracking (Baseline)
    # -------------------------------
    # All open cells except the bot's estimated spawn are possible rat positions.
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != estimated_spawn}
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = estimated_spawn
    print(f"True Rat Spawn: {true_rat_pos}")
    
    # Initialize a uniform probability distribution over rat locations.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    target_path = []
    max_steps_phase2 = 200
    
    # Move toward the rat using BFS pathfinding and update probabilities via pings.
    while bot_pos != true_rat_pos and moves < max_steps_phase2:
        if not target_path:
            if not rat_knowledge:
                print("Rat knowledge base empty, cannot find rat.")
                return moves, senses, pings, None, true_rat_pos
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                print(f"No path to {target_pos}, removing from KB")
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue
        
        direction = target_path.pop(0)
        new_pos = move(ship, bot_pos, direction)
        moves += 1
        bot_pos = new_pos
        
        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        print(f"Phase 2: Pinged at {bot_pos}, ping: {ping}, prob: {ping_prob_true:.3f}")
        
        if ping and dist_to_rat == 0:
            print(f"Rat found at {bot_pos}")
            return moves, senses, pings, None, true_rat_pos
        
        # Update rat probabilities similar to baseline.
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0
        total_prob = 0.0
        for pos in rat_knowledge:
            if pos == bot_pos:
                continue
            d = manhattan_distance(bot_pos, pos)
            p_val = 1.0 if d == 0 else np.exp(-alpha * (d - 1))
            if ping:
                rat_probs[pos] *= p_val
            else:
                rat_probs[pos] *= (1.0 - p_val)
            total_prob += rat_probs[pos]
        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob
        
        # Prune unlikely candidates.
        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 1e-4]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)
    
    print(f"Phase 2 done. Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    return moves, senses, pings, None, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, _, true_rat_pos = baseline_bot(ship)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")
