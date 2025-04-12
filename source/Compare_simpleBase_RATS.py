# compare_simple_baseline_rat_types_avg_moves.py

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
    Look up the values of the eight neighboring cells (including diagonals)
    around the given position. If a neighbor lies outside the ship boundaries,
    treat it as 0 (blocked).
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
    Given a position (row, col) and a move direction string (e.g., "up", "down"),
    calculate the new position one cell in that direction.
    """
    deltas = {'up': (-1, 0), 'down': (1, 0),
              'left': (0, -1), 'right': (0, 1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Return the opposite direction. For example, "up" becomes "down".
    Useful to backtrack or invert a sequence of moves.
    """
    rev = {'up': 'down', 'down': 'up',
           'left': 'right', 'right': 'left'}
    return rev[direction]

def move_rat(ship, rat_pos):
    """
    Attempt to move the rat in one of the four cardinal directions.
    If any direction leads to a new, valid position, randomly pick one.
    Otherwise, the rat stays still.
    """
    valid_moves = [m for m in ['up', 'down', 'left', 'right']
                   if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        return move(ship, rat_pos, random.choice(valid_moves))
    return rat_pos

# -----------------------------------------------------------------------------
# Function: simple_baseline_bot
# -----------------------------------------------------------------------------
# Purpose:
#   Implements a simple baseline approach for the bot with two phases:
#     1) Localization (no Bayesian logic).
#     2) Rat Tracking (also no Bayesian logic).
#   The bot re-targets the rat frequently (every step) and has minimal checks
#   for avoiding movement oscillations.
#
#   alpha controls ping detection probability for distance-based detection.
# -----------------------------------------------------------------------------
def simple_baseline_bot(ship, alpha=0.15, moving_rat=False):
    """
    If moving_rat=True, the rat moves each turn; otherwise, it stays put.

    Phase 1: Localization
      - The bot starts by assuming it could be in any open cell.
      - It uses sensor readings (neighbors) to prune its knowledge of possible spawns.
      - It randomly picks a valid move, updates the sensor reading, and prunes again.
      - If it ends up with exactly one candidate, it back-calculates to find
        its original spawn.

    Phase 2: Rat Tracking
      - Once the bot is localized (or forced to proceed), it tries to find
        the rat. It re-targets each step, using BFS paths to go toward
        the rat's most likely location (though here it's just a simple approach,
        not a Bayesian approach).
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # The actual spawn
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]

    # Keep track of how we moved in order to potentially backtrack.
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    # Filter possible starting locations based on the sensor reading.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
    step = 2
    max_steps_phase1 = 100

    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right']
                       if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        bot_pos = new_pos
        # Sense again after moving.
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)

        # Simulate the same move on each candidate position and check if the
        # resulting sensor reading matches the bot's real sensor reading.
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    # Check if exactly one candidate remains; if so, deduce the original spawn.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        # Reverse the move history to find the candidate's original position.
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1, None))
    else:
        final_pos = bot_pos
    bot_pos = final_pos

    # --- Phase 2: Rat Tracking ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
    max_steps = 1000

    # Attempt to handle potential oscillations by analyzing position history.
    position_history = []
    history_length = 10
    oscillation_threshold = 4

    while bot_pos != true_rat_pos and step < max_steps:
        # Log current position to detect repeating patterns like A->B->A->B
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)

        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent_positions = position_history[-oscillation_threshold:]
            # If the path is repeating every 2 moves (like A->B->A->B),
            # we consider that an oscillation.
            if (recent_positions[0] == recent_positions[2] and
                recent_positions[1] == recent_positions[3] and
                recent_positions[0] != recent_positions[1]):
                oscillating = True

        if oscillating:
            # Break the cycle by moving randomly.
            valid_moves = [m for m in ['up', 'down', 'left', 'right']
                           if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                direction = random.choice(valid_moves)
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            if not rat_knowledge:
                # We have no knowledge about where the rat could be, so quit.
                return moves, senses, pings, steps, true_rat_pos
            # The bot picks the best candidate based on rat_probs plus a tie-break
            # on distance. This code doesn't do Bayesian updates, but does re-target
            # at every step.
            target_pos = max(rat_probs, key=lambda pos: (rat_probs[pos],
                                                         -manhattan_distance(bot_pos, pos)))
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                # If there's no path, discard that candidate from the knowledge base.
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

            # Take the next step on the path toward that target.
            direction = target_path.pop(0)
            new_pos = move(ship, bot_pos, direction)
            moves += 1
            bot_pos = new_pos

        # Check if we've already caught the rat (same cell) before it moves.
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            return moves, senses, pings, steps, true_rat_pos

        # If the user selected a moving rat, let it move now.
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            # Recalculate probabilities of where the rat might be. The code below
            # does a naive approach (no real Bayesian updating).
            new_probs = {}
            for pos in rat_knowledge:
                valid_moves = [m for m in ['up', 'down', 'left', 'right']
                               if move(ship, pos, m) != pos]
                if valid_moves:
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    # Chance it stayed in place:
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    # Chance it moved to each valid neighbor:
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs
            rat_knowledge = set(new_probs.keys())

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        # If the ping is successful and the distance is zero, we've definitely caught the rat.
        if ping and dist_to_rat == 0:
            return moves, senses, pings, steps, true_rat_pos

        # If the bot is on a cell that might have been the rat's location but we didn't get a ping,
        # we set that probability to zero.
        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            return moves, senses, pings, steps, true_rat_pos

        step += 1

    return moves, senses, pings, steps, true_rat_pos


# -----------------------------------------------------------------------------
# EVALUATION AND PLOTTING LOGIC
# -----------------------------------------------------------------------------
def evaluate_bot(ship_size=30, num_trials=100):
    """
    Evaluate how many moves the simple baseline bot needs on average to 
    localize and track the rat, under different alpha values, for both 
    stationary and moving rats. We run multiple trials and average the results.
    """
    alpha_values = np.arange(0.0, 0.90, 0.05)
    stationary_avg_moves = []
    moving_avg_moves = []

    # Generate a ship of the specified size only once (if you want a new random
    # layout each alpha, generate inside the loop).
    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Evaluate with a stationary rat
        stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=False)
            stationary_moves.append(moves)
        stationary_avg = np.mean(stationary_moves)
        stationary_avg_moves.append(stationary_avg)

        # Evaluate with a moving rat
        moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=True)
            moving_moves.append(moves)
        moving_avg = np.mean(moving_moves)
        moving_avg_moves.append(moving_avg)

        # Display progress in the console
        print(f"Alpha = {alpha:.2f} done, "
              f"Avg Moves (Stationary) = {stationary_avg:.2f}, "
              f"Avg Moves (Moving) = {moving_avg:.2f}")

    return alpha_values, stationary_avg_moves, moving_avg_moves

def plot_comparison(alpha_values, stationary_avg_moves, moving_avg_moves, ship_size, num_trials):
    """
    Plots the results from evaluate_bot function. Each line represents
    how the average moves scale with alpha for stationary vs moving rats.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, stationary_avg_moves, marker='o',
             label='Simple Baseline (Stationary)', color='skyblue')
    plt.plot(alpha_values, moving_avg_moves, marker='o',
             label='Simple Baseline (Moving)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.title(
        f"Simple Baseline (Stationary vs Moving Rat)\n"
        f"Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha"
    )
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    # For quick testing, we evaluate and then plot the results.
    ship_size = 30
    num_trials = 100
    alpha_vals, stationary_avgs, moving_avgs = evaluate_bot(ship_size=ship_size,
                                                            num_trials=num_trials)
    plot_comparison(alpha_vals, stationary_avgs, moving_avgs, ship_size, num_trials)
