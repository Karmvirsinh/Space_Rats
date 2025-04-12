import numpy as np      
import random            
from ship_generator import generate_ship  
from utils import move, ping_detector, manhattan_distance, bfs_path  


# ----- Helper Functions -----

def get_surroundings(ship, pos):
    # Get the size of the grid (we assume the grid is a square).
    D = ship.shape[0]
    # These numbers represent the positions around a cell (diagonals, up, down, left, right).
    directions = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1),           (0, 1), 
                  (1, -1),  (1, 0),  (1, 1)]
    pattern = []  # This list will hold the values of the cells around our position.
    for dx, dy in directions:
        # Calculate the position of a neighboring cell.
        r, c = pos[0] + dx, pos[1] + dy
        # Check if this neighboring cell is inside the grid.
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])  # If inside, add its value.
        else:
            pattern.append(0)  # If outside the grid, treat it as blocked (0).
    return tuple(pattern)  # Return the surrounding values as a set of numbers.

def add_move(pos, direction):
    # This dictionary tells us how a move changes our position.
    deltas = {'up': (-1, 0), 'down': (1, 0), 
              'left': (0, -1), 'right': (0, 1)}
    dx, dy = deltas[direction]
    # Return the new position after moving one step in the chosen direction.
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    # This sets up the opposite move for each direction.
    rev = {'up': 'down', 'down': 'up', 
           'left': 'right', 'right': 'left'}
    # Return the opposite direction.
    return rev[direction]

def move_rat(ship, rat_pos):
    # This function helps the rat move from its current cell.
    # It creates a list of moves that actually change the rat's position.
    valid_moves = [m for m in ['up', 'down', 'left', 'right'] 
                   if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        # If there are valid moves, choose one randomly and return the new position.
        return move(ship, rat_pos, random.choice(valid_moves))
    # If no moves change the position, return the current position (rat doesn't move).
    return rat_pos

# ----- Simple Baseline Bot (No Bayesian Updates, Re-Target Every Step, Oscillation Handling, Moving Rat Option) -----

def simple_baseline_bot(ship, alpha=0.15):
    # Get the size of the grid and set initial counts for steps (moves, sensor readings, and detection pings).
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # Ask the user which type of rat to simulate.
    print("Select rat type:")
    print("1. Stationary Rat")
    print("2. Moving Rat")
    choice = input("Enter 1 or 2: ")
    moving_rat = (choice == '2')  # If the user chooses option 2, the rat will move.

    # --- Phase 1: Localization (Finding where the bot started) ---
    # We start by saying the bot could have been born in any open cell (value 1) on the grid.
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    # Randomly choose where the bot actually started.
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos  # The bot's current location.
    # Save the starting information in a list so we can review the steps later.
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]
    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    move_history = []  # We will remember each move the bot makes.
    # Look around the bot's current cell to see what cells are open or blocked.
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1  # Count that we took one sensor reading.
    # Count how many neighboring cells are blocked.
    blocked_count = sum(1 for x in current_sensor if x == 0)
    # Narrow down the possible starting positions by comparing these sensor readings.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    print(f"Step 1: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
    step = 2  # Start counting steps from here.
    max_steps_phase1 = 100  # We will only try to localize for a maximum of 100 steps.

    # Continue moving and sensing until we narrow down to one possible starting spot.
    while len(bot_knowledge) > 1 and step < max_steps_phase1:
        # Figure out which moves (up, down, left, right) actually change the bot's spot.
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] 
                       if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            print(f"Step {step}: No valid moves available, stopping")
            break  # Stop if there is nowhere valid to move.
        # Pick one of the valid moves at random.
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        # Update the bot's position by making the move.
        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1  # Count this as a move.
        bot_pos = new_pos
        # Take another sensor reading after moving.
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)
        print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensed {blocked_count} blocked neighbors")

        # Update our list of possible starting positions:
        # For each candidate position, see what happens if that move was made.
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            # If the sensor reading after the move matches what we see now, keep it.
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        print(f"Step {step}: Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    # If we end up with exactly one candidate, we can backtrack to determine the bot's start.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        # Reverse each move made during localization to go back to the start.
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print(f"Step {step}: Localized, Bot at {estimated_spawn}")
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1, None))
    else:
        # If we cannot narrow it down to one spot, we use the current position.
        final_pos = bot_pos
        print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
    print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    # --- Phase 2: Rat Tracking (Finding the rat) ---
    # Set up a list of possible positions for the rat (all open cells not at the bot's start).
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) 
                     if ship[r, c] == 1 and (r, c) != final_pos}
    # Randomly choose where the rat actually starts.
    true_rat_pos = random.choice(list(rat_knowledge))
    # Reset the bot's position to its estimated starting position.
    bot_pos = final_pos
    print(f"True Rat Spawn: {true_rat_pos}")

    # Give every possible rat position an equal chance at the start.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
    max_steps = 1000  # We will try tracking the rat for up to 1000 moves.

    # Set up storage to remember recent positions to detect if the bot is stuck in a loop.
    position_history = []
    history_length = 10  # We remember the last 10 positions.
    oscillation_threshold = 4  # We check for a repeating 4-step pattern.

    # Now, the bot will try to move until it reaches the rat.
    while bot_pos != true_rat_pos and step < max_steps:
        # Add the current bot position to our history list.
        position_history.append(bot_pos)
        if len(position_history) > history_length:
            position_history.pop(0)

        # Look for a pattern like A -> B -> A -> B in recent moves (oscillation).
        oscillating = False
        if len(position_history) >= oscillation_threshold:
            recent_positions = position_history[-oscillation_threshold:]
            if (recent_positions[0] == recent_positions[2] and
                recent_positions[1] == recent_positions[3] and
                recent_positions[0] != recent_positions[1]):
                oscillating = True

        if oscillating:
            # If the bot is stuck in a loop, choose a random move to try and get out of it.
            valid_moves = [m for m in ['up', 'down', 'left', 'right'] 
                           if move(ship, bot_pos, m) != bot_pos]
            if valid_moves:
                direction = random.choice(valid_moves)
                print(f"Step {step}: Oscillation detected, making random move {direction} to break cycle")
                bot_pos = move(ship, bot_pos, direction)
                moves += 1
        else:
            # Under normal circumstances, choose a direction to move towards where we think the rat might be.
            if not rat_knowledge:
                print("Rat knowledge base is empty, cannot find rat.")
                print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
                return moves, senses, pings, steps, true_rat_pos
            # Pick the candidate that has the highest chance (and is closer to the bot).
            target_pos = max(rat_probs, key=lambda pos: (rat_probs[pos], -manhattan_distance(bot_pos, pos)))
            # Find a path from our current position to that candidate.
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                print(f"No path to {target_pos}, removing from KB")
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

            # Take one step along that path.
            direction = target_path.pop(0)
            new_pos = move(ship, bot_pos, direction)
            moves += 1
            if new_pos != bot_pos:
                print(f"Step {step}: Moved {direction} to {new_pos}, Rat KB size: {len(rat_knowledge)}")
            bot_pos = new_pos

        # Check if we have reached the rat before the rat can move.
        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: 1.000")
            print(f"Rat found at {bot_pos}")
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        # If the rat is supposed to move, update its location.
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            if true_rat_pos != old_rat_pos:
                print(f"Step {step}: Rat moved to {true_rat_pos}")
            # Update our belief about where the rat could be by adjusting probabilities.
            new_probs = {}
            for pos in rat_knowledge:
                valid_moves = [m for m in ['up', 'down', 'left', 'right'] 
                               if move(ship, pos, m) != pos]
                if valid_moves:
                    # Calculate the chance of the rat staying or moving.
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs
            # Refresh the list of possible rat positions.
            rat_knowledge = set(new_probs.keys())

        # Ping to check if the rat is nearby.
        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: {ping_prob_true:.3f}")

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        # If a ping is heard and the bot is at the same position as the rat, we have caught it.
        if ping and dist_to_rat == 0:
            print(f"Rat found at {bot_pos}")
            print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        # If we don't hear a ping at our current position, lower the chance for this cell.
        if bot_pos in rat_probs and not ping:
            rat_probs[bot_pos] = 0
            if bot_pos in rat_knowledge:
                rat_knowledge.remove(bot_pos)

        # If our list of possible rat positions is empty, we stop searching.
        if not rat_probs:
            print("Rat knowledge base is empty, cannot find rat.")
            print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        print(f"Step {step}: Rat KB size after update: {len(rat_knowledge)}")
        step += 1

    # End of Phase 2. Print final results if we exit the loop.
    print(f"Phase 2 done, Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    print(f"Final Metrics: Moves: {moves}, Senses: {senses}, Pings: {pings}")
    return moves, senses, pings, steps, true_rat_pos

if __name__ == "__main__":
    # Create a ship of size 30x30.
    ship = generate_ship(30)
    # Run the simple baseline bot on the ship.
    moves, senses, pings, steps, true_rat_pos = simple_baseline_bot(ship)
