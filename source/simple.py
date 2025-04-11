
import numpy as np
import random
from utils import move, ping_detector, manhattan_distance, bfs_path
from ship_generator import generate_ship

def sense_neighbors(ship, pos):
    """
    A simple sensor function: counts the number of blocked neighbors (8-neighbor).
    Out-of-bound cells count as blocked.
    """
    D = ship.shape[0]
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = pos[0] + dr, pos[1] + dc
            if r < 0 or r >= D or c < 0 or c >= D or ship[r, c] == 0:
                count += 1
    return count

def add_move(pos, direction):
    """Return new position after moving from pos in given direction."""
    deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dr, dc = deltas[direction]
    return (pos[0] + dr, pos[1] + dc)

def reverse_move(direction):
    """Return the opposite move of the given direction."""
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

def baseline_bot_new(ship, alpha=0.15):
    """
    New Baseline Bot:
    
    Phase 1: Localization  
      - The bot starts with all open cells as candidates.
      - It senses its 8 neighbors (using a simple blocked-neighbor count).
      - Then it takes a random valid move (to avoid oscillation) and updates its candidate set by simulating that move.
      - Repeats until the candidate set is narrowed to one; then backtracks the moves to compute its estimated spawn.
    
    Phase 2: Rat Tracking (Stationary Rat)  
      - The bot initializes a uniform probability distribution over all open cells (except its spawn).
      - It repeatedly uses its rat sensor (ping_detector) in its current cell.
      - If the bot is in a cell that has nonzero probability and the sensor does not confirm the rat is there, it sets that cell’s probability to 0.
      - It then selects the candidate cell with the highest probability and uses BFS to plan a path toward it.
      - The bot commits to that target until it reaches it; then it re-targets.
      - The process repeats until the bot’s position equals the rat’s (sensor confirms capture).
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    ### Phase 1: Localization ###
    # All open cells are candidate spawn locations.
    candidate_set = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(candidate_set))
    bot_pos = true_bot_pos
    move_history = []
    # Use the blocked-neighbor count as the sensor reading.
    current_sensor = sense_neighbors(ship, bot_pos)
    senses += 1
    candidate_set = {pos for pos in candidate_set if sense_neighbors(ship, pos) == current_sensor}
    print("Phase 1: Localization")
    print("True bot spawn:", true_bot_pos)
    print("Initial sensor (blocked neighbor count):", current_sensor, "Candidates:", len(candidate_set))
    step = 0
    max_steps_local = 100
    while len(candidate_set) > 1 and step < max_steps_local:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        current_sensor = sense_neighbors(ship, bot_pos)
        senses += 1
        # Update candidate set by simulating the move on each candidate.
        new_candidates = set()
        for pos in candidate_set:
            new_pos = move(ship, pos, chosen_move)
            if sense_neighbors(ship, new_pos) == current_sensor:
                new_candidates.add(new_pos)
        candidate_set = new_candidates
        print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensor: {current_sensor}, Candidates left: {len(candidate_set)}")
        step += 1
    # If candidate set is narrowed to one, backtrack to get spawn; else, use current bot pos.
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

    ### Phase 2: Rat Tracking (Stationary Rat) ###
    # Initialize rat candidates: all open cells except the bot spawn.
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != estimated_spawn}
    true_rat_pos = random.choice(list(rat_candidates))
    print("Phase 2: Rat Tracking")
    print("True rat spawn:", true_rat_pos)
    # Initialize a uniform probability distribution over rat candidates.
    rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}
    
    # Set initial target as the candidate with highest probability.
    current_target = max(rat_probs, key=lambda pos: rat_probs[pos])
    target_path = bfs_path(ship, bot_pos, current_target)
    # In this simple baseline, we do not re-target until we reach the current target.
    print("Initial target (highest probability):", current_target)
    
    while bot_pos != true_rat_pos:
        # Run the sensor at the current bot cell.
        pings += 1
        sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
        # If sensor pings and bot is on the rat, capture it.
        if sensor_ping and manhattan_distance(bot_pos, true_rat_pos) == 0:
            print("Rat captured at", bot_pos)
            break
        # In a cell with non-zero probability that the rat is present, if no ping is received, set that cell's probability to 0.
        if bot_pos in rat_probs and sensor_ping == False:
            rat_probs[bot_pos] = 0
        
        # Do not change target until reached.
        if bot_pos == current_target:
            # Once reached, remove it from candidate set (if not rat) and choose the new highest probability cell.
            if bot_pos != true_rat_pos:
                rat_probs[bot_pos] = 0
            remaining = {pos: prob for pos, prob in rat_probs.items() if prob > 0}
            if not remaining:
                print("No remaining cells with non-zero probability. Terminating search.")
                break
            current_target = max(remaining, key=lambda pos: remaining[pos])
            target_path = bfs_path(ship, bot_pos, current_target)
            print("New target chosen:", current_target)
        
        # If no path exists, recalc path.
        if not target_path:
            target_path = bfs_path(ship, bot_pos, current_target)
            if not target_path:
                print("No path to target; terminating search.")
                break
        
        # Move one step along the path.
        chosen_move = target_path.pop(0)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        print(f"Moved {chosen_move} -> Bot pos: {bot_pos}")
    
    print("Phase 2 complete: Bot at", bot_pos, "True rat at", true_rat_pos)
    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, est_spawn, true_rat = baseline_bot_new(ship, alpha=0.15)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")
