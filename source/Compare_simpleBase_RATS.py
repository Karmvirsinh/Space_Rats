# compare_simple_baseline_rat_types_avg_moves.py
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

# ----- Simple Baseline Bot (No Bayesian Updates, Re-Target Every Step, Oscillation Handling) -----
def simple_baseline_bot(ship, alpha=0.15, moving_rat=False):  # Added moving_rat parameter
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]

    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        bot_pos = new_pos
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)

        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1, None))
    else:
        final_pos = bot_pos
    bot_pos = final_pos

    # --- Phase 2: Rat Tracking (No Bayesian Updates) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
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
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=lambda pos: (rat_probs[pos], -manhattan_distance(bot_pos, pos)))
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                # Only remove target_pos from rat_knowledge if it exists
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

            direction = target_path.pop(0)  # Take one step toward the target
            new_pos = move(ship, bot_pos, direction)
            moves += 1
            bot_pos = new_pos

        # Check catch before rat moves
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            return moves, senses, pings, steps, true_rat_pos

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
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        if ping and dist_to_rat == 0:
            return moves, senses, pings, steps, true_rat_pos

        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            return moves, senses, pings, steps, true_rat_pos

        step += 1

    return moves, senses, pings, steps, true_rat_pos

# ----- Plotting Logic for Both Rat Types -----
def evaluate_bot(ship_size=30, num_trials=100):
    alpha_values = np.arange(0.05, 0.75, 0.05)  # 0.05, 0.1, ..., 0.7
    stationary_avg_moves = []
    moving_avg_moves = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Evaluate for stationary rat
        stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=False)
            stationary_moves.append(moves)
        stationary_avg = np.mean(stationary_moves)
        stationary_avg_moves.append(stationary_avg)

        # Evaluate for moving rat
        moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=True)
            moving_moves.append(moves)
        moving_avg = np.mean(moving_moves)
        moving_avg_moves.append(moving_avg)

        # Simplified console output
        print(f"Alpha = {alpha:.2f} done, Avg Moves (Stationary) = {stationary_avg:.2f}, Avg Moves (Moving) = {moving_avg:.2f}")

    return alpha_values, stationary_avg_moves, moving_avg_moves

def plot_comparison(alpha_values, stationary_avg_moves, moving_avg_moves, ship_size, num_trials):
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, stationary_avg_moves, marker='o', label='Simple Baseline (Stationary)', color='skyblue')
    plt.plot(alpha_values, moving_avg_moves, marker='o', label='Simple Baseline (Moving)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(
        f"Simple Baseline: Average Moves vs Alpha (Stationary vs Moving Rat)\n"
        f"(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.show()

if __name__ == "__main__":
    ship_size = 30
    num_trials = 10
    alpha_values, stationary_avg_moves, moving_avg_moves = evaluate_bot(ship_size=ship_size, num_trials=num_trials)
    plot_comparison(alpha_values, stationary_avg_moves, moving_avg_moves, ship_size, num_trials)