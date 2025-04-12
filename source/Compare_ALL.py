# compare_simple_baseline_vs_custom_bayesian_all_rat_types.py

import matplotlib.pyplot as plt
import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_surroundings(ship, pos):
    """
    Gather the 8 neighboring cell values around 'pos'. If a neighbor cell
    is off-grid, treat it as 0.
    """
    D = ship.shape[0]
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
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
    Move from pos by one cell in the specified direction.
    """
    deltas = {'up': (-1, 0), 'down': (1, 0),
              'left': (0, -1), 'right': (0, 1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Return the opposite direction string (e.g., 'up'->'down').
    """
    rev = {'up': 'down', 'down': 'up',
           'left': 'right', 'right': 'left'}
    return rev[direction]

def move_rat(ship, rat_pos):
    """
    Try to move the rat in one random valid direction. If none are valid,
    it stays where it is.
    """
    valid_moves = [m for m in ['up', 'down', 'left', 'right']
                   if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        return move(ship, rat_pos, random.choice(valid_moves))
    return rat_pos

# -----------------------------------------------------------------------------
# SIMPLE BASELINE BOT
# -----------------------------------------------------------------------------
# No Bayesian logic, no pruning. The bot re-targets each step and tries to handle
# oscillations, but it basically just moves around randomly in Phase 1
# and does re-targeting in Phase 2.
# -----------------------------------------------------------------------------
def simple_baseline_bot(ship, alpha, moving_rat=False):
    """
    Phase 1: Basic localization (no Bayesian updates).
    Phase 2: Basic rat tracking (no Bayesian updates, re-target every step).
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # Actual position
    bot_pos = true_bot_pos
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    # Filter out all positions that don't match the sensor reading.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right']
                       if move(ship, bot_pos, m) != bot_pos]
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
        # Reverse the move sequence to find the original spawn
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking ---
    # No Bayesian logic, no pruning, re-target every step, handle potential
    # oscillations with a small cycle detection.
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != estimated_spawn}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1000

    # Use a history to detect 2-step cycles: A->B->A->B
    position_history = []
    history_length = 10
    oscillation_threshold = 4

    while bot_pos != true_rat_pos and step < max_steps:
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)

        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent = position_history[-oscillation_threshold:]
            if (recent[0] == recent[2] and
                recent[1] == recent[3] and
                recent[0] != recent[1]):
                oscillating = True

        if oscillating:
            # Break the cycle by making a random move
            valid_moves = [m for m in ['up', 'down', 'left', 'right']
                           if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                direction = random.choice(valid_moves)
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            if not rat_knowledge:
                break  # We have no info on rat positions
            # Basic logic: pick the rat position with the highest assigned probability
            # plus a negative distance factor as a tiebreaker.
            target_pos = max(rat_probs,
                             key=lambda pos: (rat_probs[pos],
                                              -manhattan_distance(bot_pos, pos)))
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

            direction = target_path.pop(0)
            bot_pos = move(ship, bot_pos, direction)
            moves += 1

        # If we land on the rat before it moves
        if bot_pos == true_rat_pos:
            # Check via ping to confirm
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if ping:
                break

        # If the rat can move
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            # Update the probabilities of each possible rat location
            # in a naive manner (no real Bayesian logic, just uniform transitions).
            new_probs = {}
            for pos in rat_knowledge:
                valid_moves = [m for m in ['up', 'down', 'left', 'right']
                               if move(ship, pos, m) != pos]
                if valid_moves:
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    # The rat might stay in place or move to one of the neighbors
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) \
                                              + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs
            rat_knowledge = set(new_probs.keys())

        # Another ping attempt
        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        # If the ping fails and we're on that cell, it can't be the rat location
        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            break

        step += 1

    return moves, senses, pings, estimated_spawn, true_rat_pos

# -----------------------------------------------------------------------------
# CUSTOM BOT (Bayesian Updates, Pruning)
# -----------------------------------------------------------------------------
# Similar structure, but tries to maintain and update a probability distribution
# (rat_probs) in a more explicit Bayesian manner. Also tries to prune improbable
# rat positions.
# -----------------------------------------------------------------------------
def custom_bot_bayesian(ship, alpha=0.15, moving_rat=False):
    """
    A more advanced bot that uses Bayesian updates and probability prunes
    for both phases: localization and rat tracking.

    Phase 1: Localization
      - Similar approach but uses sensor patterns more systematically
        to narrow possible starting positions.

    Phase 2: Rat Tracking
      - Maintains a probability distribution over all possible rat positions,
        updating it with 'ping' results. Also handles a moving rat with
        transition probabilities.
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    # Narrow down based on sensor reading from the start.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right']
                       if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1

        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        new_knowledge = set()
        # For each potential spawn position in the knowledge set, simulate the same move
        # and compare sensor readings.
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        # Reverse the recorded moves to figure out the initial spawn
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Bayesian updates) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    # Probability distribution: each open cell except the bot's cell is equally likely.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1500
    target_path = []

    while bot_pos != true_rat_pos and step < max_steps:
        # If no path is set, pick the highest-probability cell to target.
        if not target_path:
            if not rat_knowledge:
                break
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                # If no path, remove that candidate from distribution.
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

        # Take one step along the path to the chosen target.
        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

        # Check if the bot has just found the rat before it moves.
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if ping:
                break

        # If the rat is moving, let it make a move next.
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            # Update the distribution for all cells, factoring in that the rat
            # can stay still or move to neighbors with uniform probability.
            new_probs = {}
            for pos in rat_knowledge:
                valid_moves = [m for m in ['up', 'down', 'left', 'right']
                               if move(ship, pos, m) != pos]
                if valid_moves:
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) \
                                              + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs
            rat_knowledge = set(new_probs.keys())

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        # If the bot's position is in the distribution, reduce its probability
        # to 0 since we didn't confirm the rat is there unless we physically found it.
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0

        # Adjust each cell's probability based on the ping result (partial Bayesian approach).
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

        # Normalize so that the distribution sums to 1 again.
        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob

        # Prune tiny probabilities to keep the knowledge base smaller.
        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

        step += 1

    return moves, senses, pings, estimated_spawn, true_rat_pos

# -----------------------------------------------------------------------------
# COMPARISON LOGIC
# -----------------------------------------------------------------------------
def evaluate_bots(ship_size=30, num_trials=100):
    """
    Compare how many moves on average the simple_baseline_bot and custom_bot_bayesian
    each take, for both stationary and moving rats, across different alpha values.
    """
    alpha_values = np.arange(0.05, 0.90, 0.05)
    baseline_stationary_avg_moves = []
    baseline_moving_avg_moves = []
    custom_stationary_avg_moves = []
    custom_moving_avg_moves = []

    # Generate one ship for all alpha tests, so we compare consistently.
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

        # Evaluate custom bayesian bot (Stationary Rat)
        custom_stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bot_bayesian(ship, alpha=alpha, moving_rat=False)
            custom_stationary_moves.append(moves)
        custom_stationary_avg = np.mean(custom_stationary_moves)
        custom_stationary_avg_moves.append(custom_stationary_avg)

        # Evaluate custom bayesian bot (Moving Rat)
        custom_moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bot_bayesian(ship, alpha=alpha, moving_rat=True)
            custom_moving_moves.append(moves)
        custom_moving_avg = np.mean(custom_moving_moves)
        custom_moving_avg_moves.append(custom_moving_avg)

        # Print progress to the console
        print(f"Alpha = {alpha:.2f} done, "
              f"Avg Moves (Simple Baseline, Stationary) = {baseline_stationary_avg:.2f}, "
              f"Avg Moves (Simple Baseline, Moving) = {baseline_moving_avg:.2f}, "
              f"Avg Moves (Custom Bayesian, Stationary) = {custom_stationary_avg:.2f}, "
              f"Avg Moves (Custom Bayesian, Moving) = {custom_moving_avg:.2f}")

    return (alpha_values,
            baseline_stationary_avg_moves, baseline_moving_avg_moves,
            custom_stationary_avg_moves, custom_moving_avg_moves)

def plot_comparison(alpha_values,
                    baseline_stationary_avg_moves, baseline_moving_avg_moves,
                    custom_stationary_avg_moves, custom_moving_avg_moves,
                    ship_size, num_trials):
    """
    Plot lines for:
      - Simple Baseline (Stationary)
      - Simple Baseline (Moving)
      - Custom Bayesian (Stationary)
      - Custom Bayesian (Moving)
    showing how the average moves vary across alpha.
    """
    plt.figure(figsize=(10, 6))

    # Simple Baseline lines
    plt.plot(alpha_values, baseline_stationary_avg_moves,
             marker='o', linestyle='-',
             label='Simple Baseline (Stationary Rat)', color='skyblue')
    plt.plot(alpha_values, baseline_moving_avg_moves,
             marker='o', linestyle='--',
             label='Simple Baseline (Moving Rat)', color='skyblue')

    # Custom Bayesian lines
    plt.plot(alpha_values, custom_stationary_avg_moves,
             marker='o', linestyle='-',
             label='Custom Bayesian (Stationary Rat)', color='salmon')
    plt.plot(alpha_values, custom_moving_avg_moves,
             marker='o', linestyle='--',
             label='Custom Bayesian (Moving Rat)', color='salmon')

    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.title(
        f"Simple Baseline vs Custom Bayesian: Average Moves vs Alpha\n"
        f"(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)"
    )
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    (alpha_values,
     baseline_stationary_avg_moves, baseline_moving_avg_moves,
     custom_stationary_avg_moves, custom_moving_avg_moves) = evaluate_bots(
         ship_size=ship_size, num_trials=num_trials)
    plot_comparison(alpha_values,
                    baseline_stationary_avg_moves, baseline_moving_avg_moves,
                    custom_stationary_avg_moves, custom_moving_avg_moves,
                    ship_size, num_trials)
