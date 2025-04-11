# compare_custom_bayesian_rat_types_avg_moves.py
import matplotlib.pyplot as plt
import numpy as np
import random
from source.utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path
from source.ship_generator import generate_ship

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

def custom_bayesian_bot(ship, alpha=0.15, moving_rat=False):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0
    blocked_map = precompute_blocked(ship)

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
    max_steps = 100

    while len(bot_knowledge) > 1 and step < max_steps:
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

    # --- Phase 2: Rat Tracking ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))
    target_path = []
    max_steps = 1000

    while bot_pos != true_rat_pos and step < max_steps:
        if not target_path:
            if not rat_knowledge:
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                # Only remove target_pos from rat_knowledge if it exists
                if target_pos in rat_knowledge:
                    rat_knowledge.remove(target_pos)
                continue

        direction = target_path.pop(0)
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

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))

        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge), true_rat_pos))

        if ping and dist_to_rat == 0:
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
            # Update rat_knowledge to match the keys in rat_probs
            rat_knowledge = set(rat_probs.keys())

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

        step += 1

    return moves, senses, pings, steps, true_rat_pos

# ----- Plotting Logic for Both Rat Types -----
def evaluate_bot(ship_size=30, num_trials=100):
    alpha_values = np.arange(0.0, 0.90, 0.05)  # 0.05, 0.1, ..., 0.7
    stationary_avg_moves = []
    moving_avg_moves = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Evaluate for stationary rat
        stationary_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bayesian_bot(ship, alpha=alpha, moving_rat=False)
            stationary_moves.append(moves)
        stationary_avg = np.mean(stationary_moves)
        stationary_avg_moves.append(stationary_avg)

        # Evaluate for moving rat
        moving_moves = []
        for _ in range(num_trials):
            moves, _, _, _, _ = custom_bayesian_bot(ship, alpha=alpha, moving_rat=True)
            moving_moves.append(moves)
        moving_avg = np.mean(moving_moves)
        moving_avg_moves.append(moving_avg)

        # Simplified console output
        print(f"Alpha = {alpha:.2f} done, Avg Moves (Stationary) = {stationary_avg:.2f}, Avg Moves (Moving) = {moving_avg:.2f}")

    return alpha_values, stationary_avg_moves, moving_avg_moves

def plot_comparison(alpha_values, stationary_avg_moves, moving_avg_moves, ship_size, num_trials):
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, stationary_avg_moves, marker='o', label='Custom Bayesian (Stationary)', color='skyblue')
    plt.plot(alpha_values, moving_avg_moves, marker='o', label='Custom Bayesian (Moving)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(
        f"Custom Bayesian: Average Moves vs Alpha (Stationary vs Moving Rat)\n"
        f"(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.show()

if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    alpha_values, stationary_avg_moves, moving_avg_moves = evaluate_bot(ship_size=ship_size, num_trials=num_trials)
    plot_comparison(alpha_values, stationary_avg_moves, moving_avg_moves, ship_size, num_trials)