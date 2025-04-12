import numpy as np
import random
from source.utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path
from source.ship_generator import generate_ship

# =============================================================================
# Function: precompute_blocked
# -----------------------------------------------------------------------------
# Description:
#   This function goes through every cell in the "ship" grid. The ship is 
#   represented as a 2D array (NumPy array) where:
#     - 1 indicates an open/accessible cell.
#     - 0 indicates a blocked/non-accessible cell.
#
#   If a cell is open (1), we count how many of its eight neighbors (up, down, 
#   left, right, and diagonals) are blocked (0). This count is stored in a 
#   separate 2D array called "blocked_map," having the same dimensions as "ship."
#   The returned "blocked_map" can help us understand how surrounded each open 
#   cell is by blocked cells.
# -----------------------------------------------------------------------------
def precompute_blocked(ship):
    D = ship.shape[0]
    blocked_map = np.zeros((D, D), dtype=int)
    # Loop through every row (r) and column (c) in the grid.
    for r in range(D):
        for c in range(D):
            # Only proceed if the current cell is open (value 1).
            if ship[r, c] == 1:
                # sense_blocked is a helper that counts how many neighbors are blocked.
                blocked_map[r, c] = sense_blocked(ship, (r, c))
    return blocked_map

# =============================================================================
# Function: get_surroundings
# -----------------------------------------------------------------------------
# Description:
#   Takes the ship layout and a position (r, c), then looks at the 8 neighboring 
#   cells (including diagonals) around that position. It returns the value 
#   (either 0 or 1) of each neighbor as a tuple. If a neighbor would be off the 
#   grid, we substitute 0 (considering it "blocked" or out of bounds).
# -----------------------------------------------------------------------------
def get_surroundings(ship, pos):
    D = ship.shape[0]
    # The 8 possible directions around a cell.
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),          (0,1),
                  (1,-1),  (1,0),  (1,1)]
    pattern = []
    # Check each potential neighbor's position and see if it's within the grid.
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])
        else:
            # If it's off the grid, use 0.
            pattern.append(0)
    return tuple(pattern)

# =============================================================================
# Function: add_move
# -----------------------------------------------------------------------------
# Description:
#   Takes a current position (row, column) and a direction string ("up", "down", 
#   "left", or "right"), then calculates the new position by moving exactly 
#   one cell in that direction. For instance, if direction="up", row decreases by 1.
# -----------------------------------------------------------------------------
def add_move(pos, direction):
    deltas = {'up': (-1,0), 'down': (1,0),
              'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

# =============================================================================
# Function: reverse_move
# -----------------------------------------------------------------------------
# Description:
#   If you have a direction, such as "up," this function provides the opposite 
#   direction, which would be "down," and so on. Useful when backtracking 
#   (undoing a sequence of moves).
# -----------------------------------------------------------------------------
def reverse_move(direction):
    rev = {'up': 'down', 'down': 'up',
           'left': 'right', 'right': 'left'}
    return rev[direction]

# =============================================================================
# Function: move_rat
# -----------------------------------------------------------------------------
# Description:
#   Attempts to move the rat from its current position to a new one. The rat 
#   checks each of the four cardinal directions to see if moving that way 
#   actually changes its position (meaning the cell in that direction is open). 
#   If more than one direction is possible, it chooses randomly among them; 
#   otherwise, it stays put.
# -----------------------------------------------------------------------------
def move_rat(ship, rat_pos):
    # Build a list of directions that lead to a different (open) cell.
    valid_moves = [m for m in ['up', 'down', 'left', 'right']
                   if move(ship, rat_pos, m) != rat_pos]
    if valid_moves:
        return move(ship, rat_pos, random.choice(valid_moves))
    return rat_pos

# =============================================================================
# Function: custom_bayesian_bot
# -----------------------------------------------------------------------------
# Description:
#   This is the core function controlling the bot's behavior in two phases:
#
#   1) Localization Phase:
#     - The bot starts off not knowing exactly which open cell it spawned in.
#       So, it treats every open cell as a possible location.
#     - By sensing the surroundings (how many blocked neighbors it sees),
#       it narrows down the set of possible positions that match that sensor 
#       reading.
#     - The bot moves step by step, updating its own position and applying 
#       the same move to all candidate positions. After each move, it senses 
#       again, further pruning the candidate set until ideally only one 
#       candidate remains (meaning it has localized itself).
#
#   2) Rat Tracking Phase:
#     - Once localization finishes (or time runs out), the bot knows where it is.
#     - Then it tries to locate and catch the rat. It starts by assuming the rat 
#       could be in any open cell except the bot's own cell.
#     - The bot picks a likely target to move toward and updates its knowledge 
#       each step, using a "ping" mechanic that has a probability of success 
#       decreasing with distance (controlled by 'alpha').
#     - Depending on user input, the rat may move around too, complicating 
#       the search.
#
#   'alpha' is a sensitivity parameter that affects how likely a "ping" is 
#   to succeed when the bot and rat are far apart.
# -----------------------------------------------------------------------------
def custom_bayesian_bot(ship, alpha=0.15):
    # D is the dimension of the ship grid (assumes a square, D x D).
    D = ship.shape[0]
    # We track how many moves the bot has made, how many times it has used sensors, 
    # and how many times it has "pinged" trying to locate the rat.
    moves, senses, pings = 0, 0, 0

    # Use precompute_blocked to build a map of how blocked each open cell is. 
    # This data can help the bot reason about certain patterns, though in this 
    # script it mainly just demonstrates how many blocked neighbors each cell has.
    blocked_map = precompute_blocked(ship)

    # -------------------------------------------------------------------------
    # Ask the user to choose the type of rat (Stationary or Moving).
    # -------------------------------------------------------------------------
    print("Select rat type:")
    print("1. Stationary Rat")
    print("2. Moving Rat")
    choice = input("Enter 1 or 2: ")
    # If the user selects "2," that means the rat moves around each turn.
    moving_rat = (choice == '2')

    # -------------------------------------------------------------------------
    # PHASE 1: BOT LOCALIZATION
    # -------------------------------------------------------------------------
    # The bot believes it could be at any open cell initially. 
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}

    # The "true_bot_pos" is the actual position (not known to the bot).
    true_bot_pos = random.choice(list(bot_knowledge))
    # The bot starts at that true position, though conceptually it doesn't realize 
    # where exactly it is within its "knowledge set."
    bot_pos = true_bot_pos

    # "steps" is a record we keep of each action for debugging or analysis.
    # Each entry might be a tuple describing the bot's position, number of moves, etc.
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]

    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    # move_history will store the sequence of directions the bot took, so 
    # if we figure out the final position in the "knowledge set," we can 
    # backtrack to deduce the original spawn location.
    move_history = []

    # The bot checks the surroundings of its current position to see how many 
    # neighbors are open or blocked.
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    print(f"Step 1: Sensed surroundings, Senses: {senses}")  # Logging sensor usage.

    # Count how many blocked neighbors the bot currently senses.
    blocked_count = sum(1 for x in current_sensor if x == 0)
    # The bot updates its knowledge so that only those positions remain 
    # whose sensor reading (the arrangement of neighbors) matches what the bot sees.
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    print(f"Step 1: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))

    step = 2
    max_steps = 100  # We don't want infinite loops, so we limit it to 100 steps.

    # As long as more than one candidate position remains and we haven't 
    # exceeded our max steps, we keep moving around and pruning our knowledge set.
    while len(bot_knowledge) > 1 and step < max_steps:
        # List all directions that would actually change the bot's location 
        # (e.g., not blocked).
        valid_moves = [
            m for m in ['up', 'down', 'left', 'right'] 
            if move(ship, bot_pos, m) != bot_pos
        ]
        if not valid_moves:
            print(f"Step {step}: No valid moves available, stopping")
            break

        # Randomly pick one valid move direction from the set.
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        # Actually move the bot in that direction.
        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1  # We increment our move counter.
        print(f"Step {step}: Moved {chosen_move} to {new_pos}, Moves: {moves}")
        bot_pos = new_pos

        # After moving, we again sense the surroundings of the new position.
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1  # Another sensor check used.
        print(f"Step {step}: Sensed surroundings, Senses: {senses}")
        blocked_count = sum(1 for x in current_sensor if x == 0)
        print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensed {blocked_count} blocked neighbors")

        # We then simulate that same move on every possible candidate position in 
        # bot_knowledge. If, after that simulated move, the candidate's new 
        # sensor reading doesn't match what the bot actually sees, we discard it.
        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            # Compare the candidate's surroundings to the bot's actual sensor reading.
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge

        print(f"Step {step}: Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    # Once this loop finishes, if exactly one candidate remains, that means 
    # the bot has effectively localized its original spawn.
    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        # We backtrack in reverse order of the moves we took to figure out 
        # where that candidate started from.
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print(f"Step {step}: Localized, Bot at {estimated_spawn}")
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1, None))
    else:
        # If we never narrowed it down to one position, we at least know 
        # our current real position. We'll proceed with tracking from here.
        final_pos = bot_pos
        print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
    print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    # -------------------------------------------------------------------------
    # PHASE 2: RAT TRACKING
    # -------------------------------------------------------------------------
    # Now that we know where the bot ended up, we try to find the rat. 
    # We assume the rat can be at any open cell except the one the bot occupies.
    rat_knowledge = {
        (r, c) for r in range(D) for c in range(D)
        if ship[r, c] == 1 and (r, c) != final_pos
    }
    # Randomly pick a true rat spawn from these possibilities.
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = final_pos  # The bot starts searching from its final localization position.

    print(f"True Rat Spawn: {true_rat_pos}")

    # Start with equal probability for each potential rat position in rat_knowledge.
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

    # We might decide on a path to a particular cell we believe the rat is in, 
    # then follow that path step by step.
    target_path = []
    max_steps = 1000  # Another limit for the tracking phase, to avoid infinite loops.

    # We'll keep going while the bot hasn't found the rat and we haven't 
    # gone beyond the maximum steps allowed.
    while bot_pos != true_rat_pos and step < max_steps:
        # If we don't have a planned path yet, pick the cell with the highest 
        # probability in rat_probs as our next target.
        if not target_path:
            if not rat_knowledge:
                print("Rat knowledge base is empty, cannot find rat.")
                print(f"Final Metrics - Moves: {moves}, Senses: {senses}, Pings: {pings}")
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=rat_probs.get)
            # Use BFS to find a path from the bot's current position to that target.
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                print(f"No path to {target_pos}, removing from KB")
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

        # Take the first step of the path.
        direction = target_path.pop(0)
        new_pos = move(ship, bot_pos, direction)
        moves += 1
        if new_pos != bot_pos:
            print(f"Step {step}: Moved {direction} to {new_pos}, Moves: {moves}, Rat KB size: {len(rat_knowledge)}")
        bot_pos = new_pos

        # Check if we happen to land on the rat before it moves.
        if bot_pos == true_rat_pos:
            pings += 1
            print(f"Step {step}: Pinged at {bot_pos}, Pings: {pings}")
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: 1.000")
            print(f"Rat found at {bot_pos}")
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            print(f"Final Metrics - Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        # If we told the system the rat can move, let the rat take a step now.
        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            if true_rat_pos != old_rat_pos:
                print(f"Step {step}: Rat moved to {true_rat_pos}")

        # Each turn, the bot "pings" to see if it can detect the rat's presence.
        pings += 1
        print(f"Step {step}: Pinged at {bot_pos}, Pings: {pings}")
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: {ping_prob_true:.3f}")

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        # If the ping was successful AND the distance is 0, we definitely have the rat.
        if ping and dist_to_rat == 0:
            print(f"Rat found at {bot_pos}")
            print(f"Final Metrics - Moves: {moves}, Senses: {senses}, Pings: {pings}")
            return moves, senses, pings, steps, true_rat_pos

        # If we've just stood on a cell that might have been a candidate, 
        # set that probability to zero because the rat isn't there 
        # (unless we found it).
        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0

        # If the rat moves, we update the probabilities to reflect that movement 
        # (the rat can shift from one cell to a neighboring cell with some probability).
        if moving_rat:
            new_probs = {}
            for pos in rat_knowledge:
                valid_moves = [
                    m for m in ['up', 'down', 'left', 'right'] 
                    if move(ship, pos, m) != pos
                ]
                if valid_moves:
                    # We consider that the rat could stay in place or move to any valid direction.
                    transition_prob = 1.0 / (len(valid_moves) + 1)
                    new_probs[pos] = rat_probs.get(pos, 0) * transition_prob
                    for m in valid_moves:
                        next_pos = move(ship, pos, m)
                        new_probs[next_pos] = new_probs.get(next_pos, 0) + rat_probs.get(pos, 0) * transition_prob
            rat_probs = new_probs

        # Next, update the probabilities using the ping result.
        total_prob = 0.0
        for pos in rat_knowledge:
            # If the bot is exactly here, we've already set it to 0 above unless 
            # we found the rat. Skip it to avoid re-zeroing.
            if pos == bot_pos:
                continue
            dist = manhattan_distance(bot_pos, pos)
            # Probability of hearing a ping from this distance if it is indeed the rat location.
            ping_prob = 1.0 if dist == 0 else np.exp(-alpha * (dist - 1))
            if ping:
                rat_probs[pos] *= ping_prob
            else:
                rat_probs[pos] *= (1.0 - ping_prob)
            total_prob += rat_probs[pos]

        # Normalize probabilities so they sum up to 1 (assuming total_prob > 0).
        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob

        # Sometimes we prune positions that are extremely unlikely (e.g., < 0.0001)
        # to keep our knowledge base smaller.
        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

        print(f"Step {step}: Rat KB size after pruning: {len(rat_knowledge)}")
        step += 1

    # If we exit the loop, we either ran out of steps or found the rat.
    print(f"Phase 2 done, Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    print(f"Final Metrics - Moves: {moves}, Senses: {senses}, Pings: {pings}")
    return moves, senses, pings, steps, true_rat_pos

# =============================================================================
# Main Script Execution
# =============================================================================
# If this file is run directly (instead of being imported), we'll generate
# a 30x30 ship layout, then call the custom_bayesian_bot function using that ship.
if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = custom_bayesian_bot(ship)
