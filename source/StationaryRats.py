# compare_simple_baseline_vs_custom_bayesian.py
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

# ----- Simple Baseline Bot (No Bayesian Updates, No Pruning, Re-Target Every Step, Oscillation Handling) -----
def simple_baseline_bot(ship, alpha):
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
    step = 0
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
    caught = False

    # Add position history to detect oscillation
    position_history = []
    history_length = 10  # Track the last 10 positions to detect cycles
    oscillation_threshold = 4  # Look for cycles of length 4 (e.g., A -> B -> A -> B)

    while moves < max_steps:
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            pings += 1
            if ping_detector(bot_pos, true_rat_pos, alpha):
                caught = True
                break

        # Add current position to history
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)

        # Detect oscillation (e.g., A -> B -> A -> B pattern)
        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent_positions = position_history[-oscillation_threshold:]
            # Check for a simple repeating pattern like A -> B -> A -> B
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
                rat_knowledge.remove(target_pos)
                continue

            direction = target_path.pop(0)  # Take one step toward the target
            bot_pos = move(ship, bot_pos, direction)
            moves += 1

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            break

    return moves, senses, pings, estimated_spawn, true_rat_pos, caught

# ----- Custom Bot (Bayesian Updates, Pruning) -----
def custom_bot_bayesian(ship, alpha=0.15):
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
    step = 0
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
    max_steps = 1000
    caught = False
    target_path = []

    while moves < max_steps:
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            pings += 1
            if ping_detector(bot_pos, true_rat_pos, alpha):
                caught = True
                break

        if not target_path:
            if not rat_knowledge:
                break
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

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

    return moves, senses, pings, estimated_spawn, true_rat_pos, caught

# ----- Comparison Logic -----
def evaluate_bots(ship_size=30, num_trials=100):
    alpha_values = np.arange(0.0, 0.90, 0.05)  # 0.05, 0.1, ..., 0.7
    baseline_avg_moves = []
    baseline_success_rates = []
    custom_avg_moves = []
    custom_success_rates = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Evaluate simple baseline bot
        baseline_moves = []
        baseline_successes = 0
        for _ in range(num_trials):
            result = simple_baseline_bot(ship, alpha=alpha)
            moves = result[0]
            caught = result[-1]
            baseline_moves.append(moves)
            if caught:
                baseline_successes += 1
        avg_moves = np.mean(baseline_moves)
        success_rate = (baseline_successes / num_trials) * 100
        baseline_avg_moves.append(avg_moves)
        baseline_success_rates.append(success_rate)

        print(f"Simple Baseline Alpha {alpha:.2f}:")
        print(f"Average Moves: {avg_moves:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        print("---")

        # Evaluate custom bot (Bayesian)
        custom_moves = []
        custom_successes = 0
        for _ in range(num_trials):
            result = custom_bot_bayesian(ship, alpha=alpha)
            moves = result[0]
            caught = result[-1]
            custom_moves.append(moves)
            if caught:
                custom_successes += 1
        avg_moves = np.mean(custom_moves)
        success_rate = (custom_successes / num_trials) * 100
        custom_avg_moves.append(avg_moves)
        custom_success_rates.append(success_rate)

        print(f"Custom Bot (Bayesian) Alpha {alpha:.2f}:")
        print(f"Average Moves: {avg_moves:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        print("---")

    return alpha_values, baseline_avg_moves, custom_avg_moves

def plot_comparison(alpha_values, baseline_avg_moves, custom_avg_moves, ship_size, num_trials):
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, baseline_avg_moves, marker='o', label='Simple Baseline Bot (Moves)', color='skyblue')
    plt.plot(alpha_values, custom_avg_moves, marker='o', label='Custom Bot (Bayesian) (Moves)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(
        f"Simple Baseline vs Custom Bot (Bayesian): Average Moves vs Alpha\n(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.show()

if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    alpha_values, baseline_avg_moves, custom_avg_moves = evaluate_bots(ship_size=ship_size, num_trials=num_trials)
    plot_comparison(alpha_values, baseline_avg_moves, custom_avg_moves, ship_size, num_trials)