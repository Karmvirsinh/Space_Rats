# compare_simple_baseline_vs_custom_bayesian_moving_rat.py

import matplotlib.pyplot as plt  
import numpy as np               
import random                   
from ship_generator import generate_ship  
from utils import move, ping_detector, manhattan_distance, bfs_path


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
# The following functions provide utility routines to:
# - Get the surrounding cell values given a position.
# - Apply directional moves to positions.
# - Reverse moves (for backtracking).
# - Simulate the rat's movement.

def get_surroundings(ship, pos):
    """
    Given the ship grid (a 2D numpy array) and a position pos (row, column),
    this function collects the values of the eight neighboring cells surrounding pos.
    If a neighbor lies outside the grid boundaries, it is considered blocked (value=0).
    Returns the surroundings as a tuple.
    """
    D = ship.shape[0]  # Dimension of the ship (assumed square)
    # List of relative offsets to all 8 neighbors (diagonals, sides)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    pattern = []
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy  # Calculate neighbor cell coordinates
        if 0 <= r < D and 0 <= c < D:       # Check if the neighbor cell is within the grid
            pattern.append(ship[r, c])
        else:
            pattern.append(0)  # Treat out-of-bound cells as blocked (0)
    return tuple(pattern)

def add_move(pos, direction):
    """
    Given a starting position 'pos' and a direction (as a string: 'up', 'down', 'left', or 'right'),
    this function returns the new position after moving one cell in that direction.
    It is used to simulate moving backwards (reverse traversal) when backtracking.
    """
    # Define the change in coordinates for each move
    deltas = {'up': (-1, 0),
              'down': (1, 0),
              'left': (0, -1),
              'right': (0, 1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    """
    Given a move's direction, this function returns its opposite.
    This is especially useful for backtracking from a known end position back to the original start.
    For example, the reverse of 'up' is 'down' and vice versa.
    """
    rev = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left'
    }
    return rev[direction]

def move_rat(ship, rat_pos):
    """
    Simulates one step of the rat's random movement on the ship grid.
    The rat attempts to move one step in any of the four cardinal directions (up, down, left, or right).
    Only valid moves (moves that actually change the position) are chosen.
    If no valid moves are available (i.e., the rat is blocked), the rat remains in place.
    """
    # Create a list of moves for which the new position differs from the current position
    valid_moves = [m for m in ['up', 'down', 'left', 'right']
                   if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        # Randomly choose one valid move and return the new rat position
        return move(ship, rat_pos, random.choice(valid_moves))
    return rat_pos  # No valid moves found; rat remains in the same position

# -----------------------------------------------------------------------------
# Simple Baseline Bot (No Bayesian Updates, No Pruning, Re-Target Every Step, 
# Oscillation Handling)
# -----------------------------------------------------------------------------
# This bot works in two main phases:
#   Phase 1: Localization - It narrows down the possible starting positions (spawn cell)
#            of the bot by matching its sensed surroundings with candidate grid cells.
#   Phase 2: Rat Tracking - It tries to locate the moving rat by choosing target positions
#            based on a simple probability (without Bayesian updates). It re-targets at every step.
# Additionally, it checks for oscillation (repeating patterns of movement) and breaks out
# of such loops with a random move.

def simple_baseline_bot(ship, alpha, moving_rat=False):
    """
    Runs the simple baseline bot procedure on the provided ship grid.
    'alpha' parameter is used in the ping detection function.
    If 'moving_rat' is True, the rat is allowed to move during the simulation.
    Returns a tuple containing:
      - Total number of moves taken.
      - Number of sensory readings (senses).
      - Number of pings (attempts to verify the rat's position).
      - Estimated starting spawn position of the bot.
      - Final true position of the rat.
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0  # Initialize counters

    # --- Phase 1: Localization ---
    # Start with the assumption that the bot could have been spawned in any open cell (cells with value=1)
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # Randomly choose the actual spawn cell for simulation purposes
    bot_pos = true_bot_pos  # The bot's current position is its true position initially
    move_history = []  # To record all moves made during localization

    # The bot performs an initial sensor reading of its surroundings
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    # Filter possible spawn locations based on the sensor reading
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 2  # Initialize step count (phase 1 starts at step 2)
    max_steps_phase1 = 100  # Maximum allowed steps for localization phase

    # Continue moving and sensing until exactly one candidate location remains or max steps reached
    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        # Identify valid moves that change the position of the bot
        valid_moves = [m for m in ['up', 'down', 'left', 'right']
                       if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            break  # No valid moves available; exit loop
        chosen_move = random.choice(valid_moves)  # Randomly select one valid move
        move_history.append(chosen_move)  # Record the move to allow backtracking later
        bot_pos = move(ship, bot_pos, chosen_move)  # Update bot position
        moves += 1  # Increment move counter
        # After moving, perform another sensor reading
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1

        # Simulate the same move for every candidate location in the knowledge base.
        # Retain only those candidate positions whose sensor reading matches the current sensor reading.
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge  # Update the bot's candidate knowledge
        step += 1

    # After the localization phase, if only one candidate is left, backtrack through the move history
    # to estimate the original spawn position.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        # Reverse the moves to deduce the starting position.
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos  # If not narrowed down completely, assume the current position
    bot_pos = estimated_spawn  # Set the bot's position to the estimated spawn

    # --- Phase 2: Rat Tracking (No Bayesian Updates, No Pruning) ---
    # The bot now attempts to locate a moving rat.
    # Initialize knowledge about rat locations: any open cell except the bot's spawn.
    rat_knowledge = {(r, c) for r in range(D) for c in range(D)
                     if ship[r, c] == 1 and (r, c) != estimated_spawn}
    true_rat_pos = random.choice(list(rat_knowledge))  # Randomly choose the rat's actual starting position
    # Initialize equal probability for each candidate rat position
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1500  # Maximum allowed steps for tracking phase

    # For oscillation detection: record previous bot positions over recent steps.
    position_history = []
    history_length = 10  # How many past positions to store
    oscillation_threshold = 4  # Minimum sequence length to check for oscillation

    # Continue until the bot catches the rat or maximum steps reached.
    while bot_pos != true_rat_pos and step < max_steps:
        # Record the current position for oscillation checking.
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)  # Keep only the most recent positions

        # Check for a simple 2-step oscillation pattern: A->B->A->B.
        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent_positions = position_history[-oscillation_threshold:]
            if (recent_positions[0] == recent_positions[2] and
                recent_positions[1] == recent_positions[3] and
                recent_positions[0] != recent_positions[1]):
                oscillating = True

        if oscillating:
            # To break the oscillation cycle, pick a random valid move.
            valid_moves = [m for m in ['up', 'down', 'left', 'right']
                           if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                direction = random.choice(valid_moves)
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            # Select the candidate rat position with the highest probability.
            if not rat_knowledge:
                break  # Exit if no candidates remain.
            target_pos = max(rat_probs,
                             key=lambda pos: (rat_probs[pos],
                                              -manhattan_distance(bot_pos, pos)))
            # Compute a path to the chosen candidate.
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                # If no valid path exists, eliminate the candidate and try the next best.
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

            # Follow the path by taking one step along it.
            direction = target_path.pop(0)
            bot_pos = move(ship, bot_pos, direction)
            moves += 1

        # Check if bot catches the rat before the rat moves.
        if bot_pos == true_rat_pos:
            pings += 1  # Increase ping count as the bot "pings" the rat to check if it's there.
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if ping:  # If ping confirms the rat's presence, break out of loop.
                break

        # If the rat is set to be moving, update its position.
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            # Update rat movement probabilities uniformly.
            new_probs = {}
            for pos in rat_knowledge:
                # Calculate valid moves for the rat from the candidate position.
                valid_moves = [m for m in ['up','down','left','right'] if move(ship, pos, m) != pos]
                if valid_moves:
                    # Divide probability equally among staying or moving in one of the valid directions.
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) + rat_probs.get(pos, 0) * transition_prob
            # Update the probability dictionary and the candidate knowledge set.
            rat_probs = new_probs
            rat_knowledge = set(new_probs.keys())

        pings += 1  # Increase ping counter for detection attempt.
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        # If the bot stands on a cell which previously had some chance for the rat,
        # and the sensor does not detect the rat, then eliminate that candidate.
        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        if not rat_probs:
            break  # Exit if no candidate positions remain

        step += 1  # Move to the next step

    # Return the summary metrics: number of moves, number of senses, number of pings, estimated spawn, and rat's true position.
    return moves, senses, pings, estimated_spawn, true_rat_pos

# -----------------------------------------------------------------------------
# Custom Bot (Bayesian Updates, Pruning)
# -----------------------------------------------------------------------------
# This function implements a more sophisticated bot which utilizes Bayesian updates
# and pruning of unlikely candidate positions after each sensor reading. It works similarly
# to the simple bot but applies Bayesian inference during rat tracking to update belief
# distributions, and prunes candidates whose probability is too low.

def custom_bot_bayesian(ship, alpha=0.15, moving_rat=False):
    """
    Runs the custom Bayesian bot procedure on the ship.
    'alpha' is used in the ping detector probability function.
    If 'moving_rat' is True, the rat's movement is simulated at each step.
    Returns a tuple with the total number of moves, senses, pings, the estimated spawn, and the rat's true final position.
    """
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0  # Initialize counters

    # --- Phase 1: Localization ---
    # Assume possible spawn locations are any open cell on the ship.
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))  # Randomly select actual spawn for simulation
    bot_pos = true_bot_pos
    move_history = []  # Record move history for backtracking
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    # Filter candidate spawn positions based on initial sensor reading.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    step = 2  # Starting step index (to be consistent)
    max_steps_phase1 = 100  # Limit the number of localization steps

    # Continue the sensor reading and movement process until only one candidate remains
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
        # Filter candidate positions based on whether the move and resultant sensor match the bot's readings.
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        step += 1

    # Backtrack if only one candidate remains to find estimated original spawn.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos  # Fallback: use current bot position
    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Bayesian Updates, Pruning) ---
    # Initialize candidate positions for the rat (all open cells excluding the bot's position).
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))  # Choose an actual starting position for the rat
    # Uniform initial probability over all candidate positions.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 1000  # Maximum number of steps allowed for tracking
    target_path = []  # Initialize target path as empty

    # Continue searching for the rat until the bot catches the rat or the maximum steps are reached.
    while bot_pos != true_rat_pos and step < max_steps:
        if not target_path:
            # When there is no precomputed target path, choose the candidate with highest probability.
            if not rat_knowledge:
                break
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                # If no path is found, eliminate the target candidate.
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

        # Follow the computed path by taking one step.
        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

        # Check if the bot has caught the rat. If yes, confirm with a ping.
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if ping:
                break

        # If the rat is moving, update its position and adjust the probability distribution accordingly.
        if moving_rat:
            old_rat_good = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            new_probs = {}
            for pos in rat_knowledge:
                # Determine valid moves from the candidate position.
                valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, pos, m) != pos]
                if valid_moves:
                    # Calculate the transition probability accounting for staying in place as an option.
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs
            rat_knowledge = set(new_probs.keys())

        pings += 1  # Increase ping counter for rat checking
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        # Regardless of the ping reading, eliminate the candidate at the bot's current position from suspicion.
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0

        # Update the probability distribution with a Bayesian update:
        # For each candidate, adjust its probability based on the distance from the bot and whether a ping was detected.
        total_prob = 0.0
        for pos in rat_knowledge:
            if pos == bot_pos:
                continue  # Skip the bot's cell.
            dist = manhattan_distance(bot_pos, pos)
            # Calculate ping probability based on distance. If the distance is 0, then it's 100% chance.
            ping_prob = 1.0 if dist == 0 else np.exp(-alpha * (dist - 1))
            if ping:
                rat_probs[pos] *= ping_prob
            else:
                rat_probs[pos] *= (1.0 - ping_prob)
            total_prob += rat_probs[pos]

        # Normalize the probabilities so that they sum to 1.
        if total_prob > 0:
            for pos in rat_probs:
                rat_probs[pos] /= total_prob

        # Prune candidate positions that have very low probability (less than a threshold).
        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

        step += 1  # Increment step counter

    # Return performance metrics and final positions.
    return moves, senses, pings, estimated_spawn, true_rat_pos

def evaluate_bots(ship_size=30, num_trials=100):
    """
    Runs a series of trials to compare the performance of the simple baseline bot versus the custom Bayesian bot.
    It evaluates these bots (with a moving rat) over a range of alpha values.
    
    Parameters:
      - ship_size: The size of the ship grid (ship_size x ship_size).
      - num_trials: The number of simulation trials per alpha value.

    Returns:
      - alpha_values: A numpy array of tested alpha values.
      - baseline_avg_moves: Average moves required by the simple baseline bot for each alpha.
      - custom_avg_moves: Average moves required by the custom Bayesian bot for each alpha.
    """
    # Define a range of alpha values to test
    alpha_values = np.arange(0.0, 0.90, 0.05)
    baseline_avg_moves = []
    custom_avg_moves = []

    # Generate the ship grid (this ship is used for all trials in the evaluation)
    ship = generate_ship(ship_size)

    # Evaluate for each alpha value
    for alpha in alpha_values:
        # Evaluate simple baseline bot (with moving rat)
        baseline_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = simple_baseline_bot(ship, alpha=alpha, moving_rat=True)
            baseline_moves.append(moves)
        baseline_avg = np.mean(baseline_moves)
        baseline_avg_moves.append(baseline_avg)

        # Evaluate custom bayesian bot (with moving rat)
        custom_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bot_bayesian(ship, alpha=alpha, moving_rat=True)
            custom_moves.append(moves)
        custom_avg = np.mean(custom_moves)
        custom_avg_moves.append(custom_avg)

        # Print progress to console for each alpha value
        print(f"Alpha = {alpha:.2f} done, "
              f"Avg Moves (Simple Baseline) = {baseline_avg:.2f}, "
              f"Avg Moves (Custom Bayesian) = {custom_avg:.2f}")

    return alpha_values, baseline_avg_moves, custom_avg_moves

def plot_comparison(alpha_values, baseline_avg_moves, custom_avg_moves, ship_size, num_trials):
    """
    Generates and displays a matplotlib plot comparing the simple baseline bot with the custom Bayesian bot.
    The comparison is made in terms of the average number of moves needed, across different alpha values.
    
    Parameters:
      - alpha_values: The alpha values corresponding to the trials.
      - baseline_avg_moves: List of average moves for the simple baseline bot.
      - custom_avg_moves: List of average moves for the custom Bayesian bot.
      - ship_size: The size of the ship grid used.
      - num_trials: The number of trials per alpha value.
    """
    plt.figure(figsize=(10, 6))  # Set figure size
    # Plot the average moves for the simple baseline bot
    plt.plot(alpha_values, baseline_avg_moves,
             marker='o', label='Simple Baseline (Moving Rat)', color='skyblue')
    # Plot the average moves for the custom Bayesian bot
    plt.plot(alpha_values, custom_avg_moves,
             marker='o', label='Custom Bayesian (Moving Rat)', color='salmon')
    plt.xlabel("Alpha")  # Label for x-axis
    plt.ylabel("Average Number of Moves")  # Label for y-axis
    plt.grid(True)  # Enable grid lines for better readability
    plt.legend(loc='upper left')  # Add a legend in the upper left corner
    # Plot title that describes the experiment setup and parameters.
    plt.title(
        f"Simple Baseline vs Custom Bayesian: Average Moves vs Alpha (Moving Rat)\n"
        f"(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)"
    )
    plt.show()  # Display the plot

if __name__ == "__main__":
    # When running the module directly, use a default ship size of 30 and 100 trials.
    ship_size = 30
    num_trials = 100
    # Evaluate the bots to gather performance metrics over several alpha values.
    alpha_values, baseline_avg_moves, custom_avg_moves = evaluate_bots(
        ship_size=ship_size,
        num_trials=num_trials
    )
    # Generate and display the comparison plot.
    plot_comparison(alpha_values, baseline_avg_moves, custom_avg_moves, ship_size, num_trials)
