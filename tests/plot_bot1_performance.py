import numpy as np
import random
import matplotlib.pyplot as plt
from utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path
from ship_generator import generate_ship

def precompute_blocked(ship):
    D = ship.shape[0]
    blocked_map = np.zeros((D, D), dtype=int)
    for r in range(D):
        for c in range(D):
            if ship[r, c] == 1:
                blocked_map[r, c] = sense_blocked(ship, (r, c))
    return blocked_map

def get_surroundings(ship, pos):
    D = ship.shape[0]
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    pattern = []
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])
        else:
            pattern.append(0)
    return tuple(pattern)

def add_move(pos, direction):
    deltas = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
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

def baseline_bot(ship, alpha=0.15, moving_rat=False, verbose=False):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0
    blocked_map = precompute_blocked(ship)

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge), None)]
    if verbose:
        print(f"True Bot Spawn: {true_bot_pos}")
        print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    if verbose:
        print(f"Step 1: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
    step = 2
    max_steps = 100

    while len(bot_knowledge) > 1 and step < max_steps:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            if verbose:
                print(f"Step {step}: No valid moves available, stopping")
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        bot_pos = new_pos
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)
        if verbose:
            print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensed {blocked_count} blocked neighbors")

        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        if verbose:
            print(f"Step {step}: Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge), None))
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        if verbose:
            print(f"Step {step}: Localized, Bot at {estimated_spawn}")
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1, None))
    else:
        final_pos = bot_pos
        if verbose:
            print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
    if verbose:
        print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    # --- Phase 2: Rat Tracking ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = final_pos
    if verbose:
        print(f"True Rat Spawn: {true_rat_pos}")

    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
    target_path = []
    max_steps = 200

    while bot_pos != true_rat_pos and step < max_steps:
        if not target_path:
            if not rat_knowledge:
                if verbose:
                    print("Rat knowledge base is empty, cannot find rat.")
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                if verbose:
                    print(f"No path to {target_pos}, removing from KB")
                del rat_probs[target_pos]
                if target_pos in rat_knowledge:  # Check if target_pos is still in rat_knowledge
                    rat_knowledge.remove(target_pos)
                continue

        direction = target_path.pop(0)
        new_pos = move(ship, bot_pos, direction)
        moves += 1
        if new_pos != bot_pos and verbose:
            print(f"Step {step}: Moved {direction} to {new_pos}, Rat KB size: {len(rat_knowledge)}")
        bot_pos = new_pos

        if bot_pos == true_rat_pos:
            pings += 1
            ping = ping_detector(bot_pos, true_rat_pos, alpha)
            if verbose:
                print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: 1.000")
                print(f"Rat found at {bot_pos}")
            steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
            return moves, senses, pings, steps, true_rat_pos

        if moving_rat:
            old_rat_pos = true_rat_pos
            true_rat_pos = move_rat(ship, true_rat_pos)
            if true_rat_pos != old_rat_pos and verbose:
                print(f"Step {step}: Rat moved to {true_rat_pos}")

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        if verbose:
            print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: {ping_prob_true:.3f}")

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        if ping and dist_to_rat == 0:
            if verbose:
                print(f"Rat found at {bot_pos}")
            return moves, senses, pings, steps, true_rat_pos

        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0

        if moving_rat:
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

        if verbose:
            print(f"Step {step}: Rat KB size after pruning: {len(rat_knowledge)}")
        step += 1

    if verbose:
        print(f"Phase 2 done, Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    return moves, senses, pings, steps, true_rat_pos

def evaluate_and_plot(ship, alpha_values, num_trials=250):
    stationary_results = {'moves': [], 'senses': [], 'pings': []}
    moving_results = {'moves': [], 'senses': [], 'pings': []}

    for alpha in alpha_values:
        print(f"Evaluating alpha = {alpha}")
        stationary_moves, stationary_senses, stationary_pings = [], [], []
        moving_moves, moving_senses, moving_pings = [], [], []

        for _ in range(num_trials):
            # Stationary rat
            moves, senses, pings, _, _ = baseline_bot(ship, alpha=alpha, moving_rat=False, verbose=False)
            stationary_moves.append(moves)
            stationary_senses.append(senses)
            stationary_pings.append(pings)

            # Moving rat
            moves, senses, pings, _, _ = baseline_bot(ship, alpha=alpha, moving_rat=True, verbose=False)
            moving_moves.append(moves)
            moving_senses.append(senses)
            moving_pings.append(pings)

        stationary_results['moves'].append(np.mean(stationary_moves))
        stationary_results['senses'].append(np.mean(stationary_senses))
        stationary_results['pings'].append(np.mean(stationary_pings))
        moving_results['moves'].append(np.mean(moving_moves))
        moving_results['senses'].append(np.mean(moving_senses))
        moving_results['pings'].append(np.mean(moving_pings))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Moves and Pings in one plot
    plt.subplot(2, 1, 1)
    plt.plot(alpha_values, stationary_results['moves'], label='Stationary Rat - Moves', marker='o', color='blue')
    plt.plot(alpha_values, moving_results['moves'], label='Moving Rat - Moves', marker='x', color='red')
    plt.plot(alpha_values, stationary_results['pings'], label='Stationary Rat - Pings', marker='o', linestyle='--', color='blue')
    plt.plot(alpha_values, moving_results['pings'], label='Moving Rat - Pings', marker='x', linestyle='--', color='red')
    plt.xlabel('Alpha')
    plt.ylabel('Average Moves/Pings')
    plt.title('Baseline Bot Performance vs Alpha')
    plt.legend()
    plt.grid()

    # Senses in a separate plot
    plt.subplot(2, 1, 2)
    plt.plot(alpha_values, stationary_results['senses'], label='Stationary Rat - Senses', marker='o', color='blue')
    plt.plot(alpha_values, moving_results['senses'], label='Moving Rat - Senses', marker='x', color='red')
    plt.xlabel('Alpha')
    plt.ylabel('Average Senses')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ship = generate_ship(30)
    alpha_values = np.arange(0, 1.05, 0.05)
    evaluate_and_plot(ship, alpha_values, num_trials=250)