# simple_baseline_bot_standalone.py
import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions -----
def get_surroundings(ship, pos):
    D = ship.shape[0]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pattern = []
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])
        else:
            pattern.append(0)
    return tuple(pattern)

def add_move(pos, direction):
    deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

def move_rat(ship, rat_pos):
    valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        return move(ship, rat_pos, random.choice(valid_moves))
    return rat_pos

# ----- Simple Baseline Bot (No Bayesian Updates, Re-Target Every Step, Oscillation Handling, Moving Rat Option) -----
def simple_baseline_bot(ship, alpha=0.15):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # Selection for rat type
    print("Select rat type:")
    print("1. Stationary Rat")
    print("2. Moving Rat")
    choice = input("Enter 1 or 2: ")
    moving_rat = (choice == '2')

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]
    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    print(f"Step 1: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            print(f"Step {step}: No valid moves available, stopping")
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        bot_pos = new_pos
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)
        print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensed {blocked_count} blocked neighbors")

        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        print(f"Step {step}: Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print(f"Step {step}: Localized, Bot at {estimated_spawn}")
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1, None))
    else:
        final_pos = bot_pos
        print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
    print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    # --- Phase 2: Rat Tracking (No Bayesian Updates) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = final_pos
    print(f"True Rat Spawn: {true_rat_pos}")

    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
    max_steps = 1000  # Match the max steps from the reference code

    # Add position history to detect oscillation
    position_history = []
    history_length = 10  # Track the last 10 positions to detect cycles
    oscillation_threshold = 4  # Look for cycles of length 4 (e.g., A -> B -> A -> B)

    while bot_pos != true_rat_pos and step < max_steps:
        # Add current position to history
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)

        # Detect oscillation (e.g., A -> B -> A -> B pattern)
        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent_positions = position_history[-oscillation_threshold:]
            if (recent_positions[0] == recent_positions[2] and
                recent_positions[1] == recent_positions[3] and
                recent_positions[0] != recent_positions[1]):
                oscillating = True

        if oscillating:
            # If oscillating, make a random move to break the cycle
            valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                direction = random.choice(valid_moves)
                print(f"Step {step}: Oscillation detected, making random move {direction} to break cycle")
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            # Normal targeting: re-target at every step, use distance tiebreaker
            if not rat_knowledge:
                print("Rat knowledge base is empty, cannot find rat.")
                print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=lambda pos: (rat_probs[pos], -manhattan_distance(bot_pos, pos)))
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                print(f"No path to {target_pos}, removing from KB")
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

            direction = target_path.pop(0)  # Take one step toward the target
            new_pos = move(ship, bot_pos, direction)
            moves += 1
            if new_pos != bot_pos:
                print(f"Step {step}: Moved {direction} to {new_pos}, Rat KB size: {len(rat_knowledge)}")
            bot_pos = new_pos

        # Check catch before rat moves
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: 1.000")
            print(f"Rat found at {bot_pos}")
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        # Move rat if selected
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            if true_rat_pos != old_rat_pos:
                print(f"Step {step}: Rat moved to {true_rat_pos}")
            # Update probabilities for rat movement
            new_probs = {}
            for pos in rat_knowledge:
                valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, pos, m) != pos]
                if valid_moves:
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs
            # Update rat_knowledge to include new positions
            rat_knowledge = set(new_probs.keys())

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: {ping_prob_true:.3f}")

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        if ping and dist_to_rat == 0:
            print(f"Rat found at {bot_pos}")
            print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            print("Rat knowledge base is empty, cannot find rat.")
            print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        print(f"Step {step}: Rat KB size after update: {len(rat_knowledge)}")
        step += 1

    print(f"Phase 2 done, Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
    return moves, senses, pings, steps, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = simple_baseline_bot(ship)