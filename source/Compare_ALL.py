# compare_simple_baseline_vs_custom_bayesian_all_rat_types.py
import matplotlib.pyplot as plt
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

# ----- Simple Baseline Bot (No Bayesian Updates, No Pruning, Re-Target Every Step, Oscillation Handling) -----
def simple_baseline_bot(ship, alpha, moving_rat=False):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (No Bayesian Updates, No Pruning) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != estimated_spawn}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1000

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
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            # Normal targeting: re-target at every step, use distance tiebreaker
            if not rat_knowledge:
                break
            target_pos = max(rat_probs, key=lambda pos: (rat_probs[pos], -manhattan_distance(bot_pos, pos)))
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

            direction = target_path.pop(0)  # Take one step toward the target
            bot_pos = move(ship, bot_pos, direction)
            moves += 1

        # Check catch before rat moves
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if ping:
                break

        # Move rat if selected
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
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

        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            break

        step += 1

    return moves, senses, pings, estimated_spawn, true_rat_pos

# ----- Custom Bot (Bayesian Updates, Pruning) -----
def custom_bot_bayesian(ship, alpha=0.15, moving_rat=False):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Bayesian Updates, Pruning) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1500
    target_path = []

    while bot_pos != true_rat_pos and step < max_steps:
        if not target_path:
            if not rat_knowledge:
                break
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

        # Check catch before rat moves
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if ping:
                break

        # Move rat if selected
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
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

        step += 1

    return moves, senses, pings, estimated_spawn, true_rat_pos

# ----- Comparison Logic -----
def evaluate_bots(ship_size=30, num_trials=100):
    alpha_values = np.arange(0.05, 0.90, 0.05)  # 0.05, 0.1, ..., 0.7
    baseline_stationary_avg_moves = []
    baseline_moving_avg_moves = []
    custom_stationary_avg_moves = []
    custom_moving_avg_moves = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Evaluate simple baseline bot (Stationary Rat)
        baseline_stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=False)
            baseline_stationary_moves.append(moves)
        baseline_stationary_avg = np.mean(baseline_stationary_moves)
        baseline_stationary_avg_moves.append(baseline_stationary_avg)

        # Evaluate simple baseline bot (Moving Rat)
        baseline_moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=True)
            baseline_moving_moves.append(moves)
        baseline_moving_avg = np.mean(baseline_moving_moves)
        baseline_moving_avg_moves.append(baseline_moving_avg)

        # Evaluate custom bot (Bayesian, Stationary Rat)
        custom_stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bot_bayesian(ship, alpha=alpha, moving_rat=False)
            custom_stationary_moves.append(moves)
        custom_stationary_avg = np.mean(custom_stationary_moves)
        custom_stationary_avg_moves.append(custom_stationary_avg)

        # Evaluate custom bot (Bayesian, Moving Rat)
        custom_moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bot_bayesian(ship, alpha=alpha, moving_rat=True)
            custom_moving_moves.append(moves)
        custom_moving_avg = np.mean(custom_moving_moves)
        custom_moving_avg_moves.append(custom_moving_avg)

        # Simplified console output
        print(f"Alpha = {alpha:.2f} done, "
              f"Avg Moves (Simple Baseline, Stationary) = {baseline_stationary_avg:.2f}, "
              f"Avg Moves (Simple Baseline, Moving) = {baseline_moving_avg:.2f}, "
              f"Avg Moves (Custom Bayesian, Stationary) = {custom_stationary_avg:.2f}, "
              f"Avg Moves (Custom Bayesian, Moving) = {custom_moving_avg:.2f}")

    return (alpha_values, baseline_stationary_avg_moves, baseline_moving_avg_moves,
            custom_stationary_avg_moves, custom_moving_avg_moves)

def plot_comparison(alpha_values, baseline_stationary_avg_moves, baseline_moving_avg_moves,
                    custom_stationary_avg_moves, custom_moving_avg_moves, ship_size, num_trials):
    plt.figure(figsize=(10, 6))
    # Simple Baseline Bot: Skyblue, solid for stationary, dashed for moving
    plt.plot(alpha_values, baseline_stationary_avg_moves, marker='o', linestyle='-',
             label='Simple Baseline (Stationary Rat)', color='skyblue')
    plt.plot(alpha_values, baseline_moving_avg_moves, marker='o', linestyle='--',
             label='Simple Baseline (Moving Rat)', color='skyblue')
    # Custom Bayesian Bot: Salmon, solid for stationary, dashed for moving
    plt.plot(alpha_values, custom_stationary_avg_moves, marker='o', linestyle='-',
             label='Custom Bayesian (Stationary Rat)', color='salmon')
    plt.plot(alpha_values, custom_moving_avg_moves, marker='o', linestyle='--',
             label='Custom Bayesian (Moving Rat)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(
        f"Simple Baseline vs Custom Bayesian: Average Moves vs Alpha\n"
        f"(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.show()

if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    (alpha_values, baseline_stationary_avg_moves, baseline_moving_avg_moves,
     custom_stationary_avg_moves, custom_moving_avg_moves) = evaluate_bots(ship_size=ship_size, num_trials=num_trials)
    plot_comparison(alpha_values, baseline_stationary_avg_moves, baseline_moving_avg_moves,
                    custom_stationary_avg_moves, custom_moving_avg_moves, ship_size, num_trials)