# compare_custom_bayesian_rat_types_avg_moves.py

import matplotlib.pyplot as plt
import numpy as np
import random
from source.utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path
from source.ship_generator import generate_ship

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def precompute_blocked(ship):
    """
    For each open cell in the grid, count how many of its eight neighbors
    are blocked. Store these counts in a 'blocked_map' of the same dimension.
    """
    D = ship.shape[0]
    blocked_map = np.zeros((D, D), dtype=int)
    for r in range(D):
        for c in range(D):
            if ship[r, c] == 1:
                blocked_map[r, c] = sense_blocked(ship, (r, c))
    return blocked_map

def get_surroundings(ship, pos):
    """
    Return the 8-cell neighbor pattern around 'pos'; if neighbor off-grid, use 0.
    """
    D = ship.shape[0]
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),          (0,1),
                  (1,-1), (1,0),   (1,1)]
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
    Shift (row, col) by one cell in the specified direction (up/down/left/right).
    """
    deltas = {'up': (-1,0), 'down': (1,0),
              'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Provide the opposite of a given move (e.g., 'up' -> 'down').
    """
    rev = {'up': 'down', 'down': 'up',
           'left': 'right', 'right': 'left'}
    return rev[direction]

def move_rat(ship, rat_pos):
    """
    Attempt to move the rat to a random valid neighboring cell among the four
    cardinal directions. If no move is valid, the rat remains where it is.
    """
    valid_moves = [m for m in ['up', 'down', 'left', 'right']
                   if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        return move(ship, rat_pos, random.choice(valid_moves))
    return rat_pos

# -----------------------------------------------------------------------------
# FUNCTION: custom_bayesian_bot
# -----------------------------------------------------------------------------
# Purpose:
#   This bot does two phases (localization and rat tracking) with a Bayesian
#   update approach. The user can optionally choose a 'moving_rat' scenario.
#
#   alpha is used in 'ping_detector' to control how detection probability
#   decreases with distance.
# -----------------------------------------------------------------------------
def custom_bayesian_bot(ship, alpha=0.15, moving_rat=False):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # Precompute how blocked each open cell is, though we might not explicitly use it here.
    blocked_map = precompute_blocked(ship)

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # Actual spawn for the bot
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]

    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    # Filter the knowledge set by matching sensor reading
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
    step = 2
    max_steps = 100

    # Use repeated sensing and movement to narrow down the spawn location.
    while len(bot_knowledge) > 1 and step < max_steps:
        valid_moves = [m for m in ['up', 'down', 'left', 'right']
                       if move(ship, bot_pos, m) != bot_pos]
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

        # Check each candidate position, see if a simulated move leads to
        # the same new sensor reading.
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    # If exactly one candidate remains, backtrack to find the original spawn.
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

    # --- Phase 2: Rat Tracking ---
    # Initially, each open cell (except the bot's) is equally likely to have the rat.
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
    target_path = []
    max_steps = 1000

    while bot_pos != true_rat_pos and step < max_steps:
        # If we don't have a path, pick the highest probability location to approach.
        if not target_path:
            if not rat_knowledge:
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

        # Take one step on the path
        direction = target_path.pop(0)
        new_pos = move(ship, bot_pos, direction)
        moves += 1
        bot_pos = new_pos

        # Check if the bot landed on the rat
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            return moves, senses, pings, steps, true_rat_pos

        # If the rat moves
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        # If the ping was successful and distance is zero, we've got the rat.
        if ping and dist_to_rat == 0:
            return moves, senses, pings, steps, true_rat_pos

        # Probability of the rat being in the bot's cell is set to zero if we haven't found it there.
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0

        # If the rat can move, recalculate the distribution for all possible positions.
        # Then update the distribution based on ping outcome. This is a partial
        # Bayesian update approach (though not fully rigorous).
        if moving_rat:
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
            rat_knowledge = set(rat_probs.keys())

        total_prob = 0.0
        for pos in rat_knowledge:
            if pos == bot_pos:
                continue
            dist = manhattan_distance(bot_pos, pos)
            ping_prob = 1.0 if dist == 0 else np.exp(-alpha * (dist - 1))
            # If ping is true, cells further away are less likely; if ping is false,
            # cells further away are more likely. This is a partial Bayesian approach.
            if ping:
                rat_probs[pos] *= ping_prob
            else:
                rat_probs[pos] *= (1.0 - ping_prob)
            total_prob += rat_probs[pos]

        # Normalize the distribution if there's some total probability.
        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob

        # Prune extremely low-probability cells.
        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

        step += 1

    return moves, senses, pings, steps, true_rat_pos

# -----------------------------------------------------------------------------
# PLOTTING LOGIC
# -----------------------------------------------------------------------------
def evaluate_bot(ship_size=30, num_trials=100):
    """
    Evaluate the custom_bayesian_bot with stationary vs moving rats across a 
    range of alpha values. We track how many moves are needed on average.
    """
    alpha_values = np.arange(0.0, 0.90, 0.05)
    stationary_avg_moves = []
    moving_avg_moves = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Stationary rat
        stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bayesian_bot(ship, alpha=alpha, moving_rat=False)
            stationary_moves.append(moves)
        stationary_avg = np.mean(stationary_moves)
        stationary_avg_moves.append(stationary_avg)

        # Moving rat
        moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bayesian_bot(ship, alpha=alpha, moving_rat=True)
            moving_moves.append(moves)
        moving_avg = np.mean(moving_moves)
        moving_avg_moves.append(moving_avg)

        print(f"Alpha = {alpha:.2f} done, "
              f"Avg Moves (Stationary) = {stationary_avg:.2f}, "
              f"Avg Moves (Moving) = {moving_avg:.2f}")

    return alpha_values, stationary_avg_moves, moving_avg_moves

def plot_comparison(alpha_values, stationary_avg_moves, moving_avg_moves, ship_size, num_trials):
    """
    Plot how many moves it takes (on average) for the custom_bayesian_bot
    to find the rat, for both stationary and moving rat scenarios, as alpha changes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, stationary_avg_moves, marker='o',
             label='Custom Bayesian (Stationary)', color='skyblue')
    plt.plot(alpha_values, moving_avg_moves, marker='o',
             label='Custom Bayesian (Moving)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.title(
        f"Custom Bayesian Bot: Stationary vs Moving Rat\n"
        f"Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha"
    )
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    alpha_vals, stationary_avgs, moving_avgs = evaluate_bot(ship_size=ship_size,
                                                            num_trials=num_trials)
    plot_comparison(alpha_vals, stationary_avgs, moving_avgs, ship_size, num_trials)
