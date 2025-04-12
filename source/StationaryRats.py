# compare_simple_baseline_vs_custom_bayesian.py

import matplotlib.pyplot as plt  
import numpy as np               
import random                  
from ship_generator import generate_ship  
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions -----

def get_surroundings(ship, pos):
    """
    Collects and returns the values of the eight neighboring cells around the given 'pos'
    in the ship grid. If a neighbor is off the grid, it appends 0 (indicating a blocked cell).
    
    Parameters:
      - ship: 2D numpy array representing the grid.
      - pos: Tuple (row, column) representing a position in the grid.
      
    Returns:
      - A tuple representing the values of the eight surrounding cells.
    """
    D = ship.shape[0]  # Dimension of the ship grid (assumed square).
    # Define relative positions for all 8 neighboring cells (diagonals and sides).
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    pattern = []  # To store values of neighboring cells.
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy  # Compute neighbor's coordinates.
        # If within bounds, append cell value; otherwise, append 0.
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])
        else:
            pattern.append(0)
    return tuple(pattern)

def add_move(pos, direction):
    """
    Moves from the given position 'pos' one step in the specified direction.
    
    Parameters:
      - pos: Tuple (row, column) of the current position.
      - direction: String indicating direction, one of 'up', 'down', 'left', 'right'.
    
    Returns:
      - A new position tuple after applying the move.
    """
    # Define movement deltas for each direction.
    deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Provides the reverse or opposite of the given directional move.
    
    Parameters:
      - direction: A string representing the move direction.
    
    Returns:
      - A string representing the opposite direction.
    """
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

# ----- Simple Baseline Bot (No Bayesian Updates, No Pruning, Re-Target Every Step, Oscillation Handling) -----

def simple_baseline_bot(ship, alpha):
    """
    Simulates the simple baseline bot which attempts to localize itself and track a rat
    without utilizing Bayesian updates or pruning. The bot re-targets at every step and
    detects oscillation patterns (like A -> B -> A -> B) to break out of them.
    
    The function works in two phases:
      1. Localization: Narrow down the bot's spawn location based on sensor readings.
      2. Rat Tracking: Move towards the rat (using a simple probability and distance tie-breaker).

    Parameters:
      - ship: The grid representing the ship.
      - alpha: Parameter used in the ping detection function.
    
    Returns:
      - moves: Number of moves taken.
      - senses: Number of sensor readings made.
      - pings: Number of detection attempts (pings).
      - estimated_spawn: The estimated starting position of the bot.
      - true_rat_pos: The true position of the rat at termination.
      - caught: Boolean indicating whether the rat was caught.
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0  # Initialize counters.

    # --- Phase 1: Localization ---
    # Start by assuming the bot may have spawned anywhere an open cell (value 1) exists.
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # Random actual spawn.
    bot_pos = true_bot_pos  # Bot's starting position.
    move_history = []  # To keep track of moves made during localization.
    current_sensor = get_surroundings(ship, bot_pos)  # Initial sensor reading.
    senses += 1
    # Narrow down candidate locations by comparing sensor readings.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 0  # Initialize step counter.
    max_steps_phase1 = 100  # Maximum allowed steps for localization.

    # Loop until a single candidate remains or maximum steps reached.
    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        # Find moves that actually change the bot's position.
        valid_moves = [m for m in ['up', 'down', 'left', 'right']
                       if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break  # Break if no valid moves.
        chosen_move = random.choice(valid_moves)  # Randomly pick a move.
        move_history.append(chosen_move)  # Record the move.
        bot_pos = move(ship, bot_pos, chosen_move)  # Update bot position.
        moves += 1  # Increment move counter.
        current_sensor = get_surroundings(ship, bot_pos)  # New sensor reading after move.
        senses += 1
        new_knowledge = set()  # For updating candidate positions.
        # Simulate the move for every candidate and filter based on sensor reading.
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge  # Update knowledge.
        step += 1

    # If a unique candidate remains, backtrack to estimate the spawn.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos
    # Reset bot's current position to estimated spawn.
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (No Bayesian Updates, No Pruning) ---
    # Initialize candidate cells for the rat (any open cell different from bot's spawn).
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != estimated_spawn}
    true_rat_pos = random.choice(list(rat_knowledge))  # Select a random true rat position.
    # Initialize uniform probability distribution over candidate rat positions.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1000  # Maximum moves allowed for tracking.
    caught = False  # Indicator for rat capture.

    # Initialize a list for recent positions to detect oscillations.
    position_history = []
    history_length = 10  # Keep the last 10 positions.
    oscillation_threshold = 4  # Check for cycles over 4 positions (e.g., A -> B -> A -> B).

    # Continue until we have exhausted moves or caught the rat.
    while moves < max_steps:
        # Check if the bot and rat are at the same position.
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            pings += 1
            if ping_detector(bot_pos, true_rat_pos, alpha):
                caught = True
                break  # Stop if the rat is caught.

        # Update position history for oscillation detection.
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)

        # Detect oscillation: if the last 4 positions form an A -> B -> A -> B pattern.
        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent_positions = position_history[-oscillation_threshold:]
            if (recent_positions[0] == recent_positions[2] and
                recent_positions[1] == recent_positions[3] and
                recent_positions[0] != recent_positions[1]):
                oscillating = True

        if oscillating:
            # If in an oscillation loop, take a random valid move to break the cycle.
            valid_moves = [m for m in ['up', 'down', 'left', 'right']
                           if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                direction = random.choice(valid_moves)
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            # In normal mode, re-target the rat at each step.
            if not rat_knowledge:
                break  # Exit if no candidate positions remain.
            # Select target candidate with highest probability and shorter Manhattan distance.
            target_pos = max(rat_probs, key=lambda pos: (rat_probs[pos],
                                                         -manhattan_distance(bot_pos, pos)))
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                # Eliminate candidate if no path exists.
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

            # Move one step along the computed path toward the target.
            direction = target_path.pop(0)
            bot_pos = move(ship, bot_pos, direction)
            moves += 1

        # Increment ping count and check if the rat is detected.
        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        # If at a candidate with no ping, eliminate that possibility.
        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        # If no candidate probabilities remain, exit the loop.
        if not rat_probs:
            break

    # Return key metrics and statuses.
    return moves, senses, pings, estimated_spawn, true_rat_pos, caught

# ----- Custom Bot (Bayesian Updates, Pruning) -----

def custom_bot_bayesian(ship, alpha=0.15):
    """
    Simulates a more advanced bot that uses Bayesian updates and candidate pruning during rat tracking.
    The bot localizes itself similarly to the simple bot, but during tracking,
    it updates the belief distribution of rat positions based on sensor readings.
    
    Parameters:
      - ship: The grid representing the ship.
      - alpha: Parameter used for calculating ping probabilities.
    
    Returns:
      - moves: Total number of moves taken.
      - senses: Total number of sensor readings performed.
      - pings: Total number of ping detection attempts.
      - estimated_spawn: The estimated starting position of the bot.
      - true_rat_pos: The true final position of the rat.
      - caught: Boolean that indicates if the rat was caught.
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # --- Phase 1: Localization ---
    # Assume potential bot positions include any open cell.
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # Choose an actual spawn location.
    bot_pos = true_bot_pos
    move_history = []  # Record movement history for backtracking.
    current_sensor = get_surroundings(ship, bot_pos)  # Get initial sensor reading.
    senses += 1
    # Filter possible spawn locations based on the sensor.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 0
    max_steps_phase1 = 100

    # Loop to narrow down possible spawn locations.
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

    # Backtrack via move history if a unique candidate was found.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Bayesian Updates, Pruning) ---
    # Define candidate rat positions (all open cells except the bot's position).
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))  # Choose actual rat's starting position.
    # Initialize uniform probability for each candidate rat position.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1000
    caught = False  # Track if rat is caught.
    target_path = []  # Holds the current path toward the target candidate.

    # Iterate until the maximum steps are reached or the rat is caught.
    while moves < max_steps:
        # If bot reaches the rat, check with a ping.
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            pings += 1
            if ping_detector(bot_pos, true_rat_pos, alpha):
                caught = True
                break

        if not target_path:
            # When no target path is computed, select the candidate with the highest probability.
            if not rat_knowledge:
                break
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                # Remove candidate if no path exists.
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

        # Move one step along the computed path.
        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

        # Update ping counter and check detection.
        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        # Set probability of bot's current cell to 0.
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0

        # Apply Bayesian update across candidate rat positions.
        total_prob = 0.0
        for pos in rat_knowledge:
            if pos == bot_pos:
                continue
            dist = manhattan_distance(bot_pos, pos)
            # Calculate likelihood via a probability that decays with distance.
            ping_prob = 1.0 if dist == 0 else np.exp(-alpha * (dist - 1))
            if ping:
                rat_probs[pos] *= ping_prob
            else:
                rat_probs[pos] *= (1.0 - ping_prob)
            total_prob += rat_probs[pos]

        # Normalize the probability distribution.
        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob

        # Prune candidates whose probability falls below a small threshold.
        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

    # Return key metrics.
    return moves, senses, pings, estimated_spawn, true_rat_pos, caught

# ----- Comparison Logic -----

def evaluate_bots(ship_size=30, num_trials=100):
    """
    Compares the performance of the simple baseline bot and the custom Bayesian bot
    over a range of alpha values. For each alpha, the function simulates a number of
    trials and computes the average number of moves and the success rate of catching the rat.
    
    Parameters:
      - ship_size: Size of the ship grid (ship_size x ship_size).
      - num_trials: Number of simulation trials per alpha value.
    
    Returns:
      - alpha_values: Array of tested alpha values.
      - baseline_avg_moves: Average moves for the simple baseline bot at each alpha.
      - custom_avg_moves: Average moves for the custom Bayesian bot at each alpha.
    """
    alpha_values = np.arange(0.0, 0.90, 0.05)  # Test alpha values: 0.0, 0.05, 0.1, ..., 0.85.
    baseline_avg_moves = []
    baseline_success_rates = []
    custom_avg_moves = []
    custom_success_rates = []

    ship = generate_ship(ship_size)  # Generate the ship grid for trials.

    for alpha in alpha_values:
        # Evaluate the simple baseline bot.
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

        # Print metrics for each alpha value for the simple bot.
        print(f"Simple Baseline Alpha {alpha:.2f}:")
        print(f"Average Moves: {avg_moves:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        print("---")

        # Evaluate the custom Bayesian bot.
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

        # Print metrics for each alpha value for the custom bot.
        print(f"Custom Bot (Bayesian) Alpha {alpha:.2f}:")
        print(f"Average Moves: {avg_moves:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        print("---")

    return alpha_values, baseline_avg_moves, custom_avg_moves

def plot_comparison(alpha_values, baseline_avg_moves, custom_avg_moves, ship_size, num_trials):
    """
    Plots a comparison graph of the average number of moves required by both the
    simple baseline bot and the custom Bayesian bot across different alpha values.
    
    Parameters:
      - alpha_values: Tested alpha values.
      - baseline_avg_moves: Average moves for the simple baseline bot.
      - custom_avg_moves: Average moves for the custom Bayesian bot.
      - ship_size: Size of the ship grid.
      - num_trials: Number of trials conducted per alpha value.
    """
    plt.figure(figsize=(10, 6))
    # Plot the simple baseline bot's performance.
    plt.plot(alpha_values, baseline_avg_moves, marker='o', label='Simple Baseline Bot (Moves)', color='skyblue')
    # Plot the custom Bayesian bot's performance.
    plt.plot(alpha_values, custom_avg_moves, marker='o', label='Custom Bot (Bayesian) (Moves)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.grid(True)
    plt.legend(loc='upper left')
    # Set the plot title with experimental details.
    plt.title(
        f"Simple Baseline vs Custom Bot (Bayesian): Average Moves vs Alpha\n(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)"
    )
    plt.show()

if __name__ == "__main__":
    # Run the experiment with default ship size and number of trials.
    ship_size = 30
    num_trials = 100
    alpha_values, baseline_avg_moves, custom_avg_moves = evaluate_bots(ship_size=ship_size, num_trials=num_trials)
    # Generate the comparison plot.
    plot_comparison(alpha_values, baseline_avg_moves, custom_avg_moves, ship_size, num_trials)
