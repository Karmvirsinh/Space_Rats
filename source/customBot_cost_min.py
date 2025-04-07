# custom_bot_cost_minimization.py
import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions for Phase 1 (Localization) -----
def get_surroundings(ship, pos):
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
    deltas = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

# ----- Custom Bot with Cost Minimization (Stationary Rat) -----
def custom_bot_cost_minimization(ship, alpha=0.15, verbose=True):
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0
    steps = []  # To track step-by-step state for logging

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
    if verbose:
        print(f"True Bot Spawn: {true_bot_pos}")
        print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    if verbose:
        print(f"Step 1: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            if verbose:
                print(f"Step {step}: No valid moves available, stopping")
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)
        if verbose:
            print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensed {blocked_count} blocked neighbors")
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        if verbose:
            print(f"Step {step}: Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        if verbose:
            print(f"Step {step}: Localized, Bot at {estimated_spawn}")
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1))
    else:
        final_pos = bot_pos
        if verbose:
            print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
    if verbose:
        print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    bot_pos = final_pos

    # --- Phase 2: Rat Tracking (Stationary Rat with Cost Minimization) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    if verbose:
        print(f"True Rat Spawn: {true_rat_pos}")
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))
    target_path = []
    max_steps = 1000

    while moves < max_steps:
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
            pings += 1
            if sensor_ping:
                if verbose:
                    print(f"Rat found at {bot_pos}")
                break

        if not target_path:
            if not rat_knowledge:
                if verbose:
                    print("Rat knowledge base is empty, cannot find rat.")
                break
            # Compute weighted average distance for each possible target
            best_target = None
            min_expected_cost = float('inf')
            for target in rat_knowledge:
                path_to_target = bfs_path(ship, bot_pos, target)
                if not path_to_target:
                    if verbose:
                        print(f"No path to {target}, removing from KB")
                    del rat_probs[target]
                    rat_knowledge.remove(target)
                    continue
                # Expected cost = sum(P(pos) * ManhattanDistance(target, pos))
                expected_cost = 0.0
                for pos in rat_knowledge:
                    dist = manhattan_distance(target, pos)
                    expected_cost += rat_probs[pos] * dist
                if expected_cost < min_expected_cost:
                    min_expected_cost = expected_cost
                    best_target = target
                    target_path = path_to_target

            if not best_target:
                # Fallback: pick a random valid move if no path to any target
                valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
                if not valid_moves:
                    if verbose:
                        print(f"Step {step}: No valid moves available, stopping")
                    break
                direction = random.choice(valid_moves)
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
                if verbose:
                    print(f"Step {step}: Moved {direction} to {bot_pos}, Rat KB size: {len(rat_knowledge)}")
                steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))
                step += 1
                continue

        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1
        if verbose:
            print(f"Step {step}: Moved {direction} to {bot_pos}, Rat KB size: {len(rat_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        if verbose:
            print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: {ping_prob_true:.3f}")

        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0
        total_prob = 0.0
        for pos in rat_knowledge:
            if pos == bot_pos:
                continue
            dist = manhattan_distance(bot_pos, pos)
            ping_prob = 1.0 if dist == 0 else np.exp(-alpha * (dist - 1))
            if ping:
                rat_probs[pos] *= ping_prob
            else:
                rat_probs[pos] *= (1.0 - ping_prob)
            total_prob += rat_probs[pos]

        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob

        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

        if verbose:
            print(f"Step {step}: Rat KB size after pruning: {len(rat_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))
        step += 1

    if verbose:
        print(f"Phase 2 done, Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    return moves, senses, pings, steps, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = custom_bot_cost_minimization(ship)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")