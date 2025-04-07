# compare_bots.py
import matplotlib.pyplot as plt
import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path


# ----- Helper Functions for Phase 1 (Localization) -----
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


# ----- Helper Functions for Phase 2 (Rat Tracking) -----
def get_cluster(pos, D, cluster_size):
    """Assign a cell to a cluster based on its position (grid-based clustering)."""
    r, c = pos
    cluster_row = r // cluster_size
    cluster_col = c // cluster_size
    clusters_per_row = D // cluster_size
    return cluster_row * clusters_per_row + cluster_col


def get_cluster_representative(cluster_cells, rat_probs):
    """Choose a representative cell for the cluster (highest probability)."""
    if not cluster_cells:
        return None
    return max(cluster_cells, key=lambda pos: rat_probs[pos])


def argmax_candidate(prob_dict):
    return max(prob_dict, key=prob_dict.get)


# ----- Baseline Bot (Stationary Rat) -----
def baseline_bot(ship, alpha=0.15, verbose=False):
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}

    max_steps_phase1 = 100
    step = 0
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
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos

    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Stationary Rat) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 150
    target_path = []
    caught = False

    while moves < max_steps:
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
            pings += 1
            if sensor_ping:
                caught = True
                break

        if not target_path:
            if not rat_knowledge:
                break
            target_pos = argmax_candidate(rat_probs)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0
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

    return moves, senses, pings, estimated_spawn, true_rat_pos, caught


# ----- Custom Bot with Improved Area Segmentation (Stationary Rat) -----
def custom_bot_area_segmentation(ship, alpha=0.15, verbose=False):
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --- Phase 1: Localization ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}

    max_steps_phase1 = 100
    step = 0
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
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = bot_pos

    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Stationary Rat with Improved Area Segmentation) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    max_steps = 300
    target_path = []
    current_cluster = None
    caught = False

    # Dynamic clustering: divide the ship into a grid of clusters
    cluster_size = 10  # Each cluster is 10x10 (for a 30x30 ship, this gives 9 clusters)
    num_clusters = (D // cluster_size) ** 2

    while moves < max_steps:
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
            pings += 1
            if sensor_ping:
                caught = True
                break

        if not target_path:
            if not rat_knowledge:
                break
            # Dynamic clustering: assign cells to clusters and compute total probability per cluster
            cluster_probs = [0.0] * num_clusters
            cluster_cells = [[] for _ in range(num_clusters)]
            for pos in rat_knowledge:
                cluster = get_cluster(pos, D, cluster_size)
                cluster_probs[cluster] += rat_probs[pos]
                cluster_cells[cluster].append(pos)

            # Aggressive pruning: remove clusters with very low total probability
            clusters_to_remove = []
            for cluster in range(num_clusters):
                if cluster_probs[cluster] < 0.01:
                    clusters_to_remove.append(cluster)
            for cluster in clusters_to_remove:
                for pos in cluster_cells[cluster][:]:
                    del rat_probs[pos]
                    rat_knowledge.remove(pos)
                cluster_cells[cluster] = []
                cluster_probs[cluster] = 0.0

            # Choose the cluster with the shortest path to its representative cell
            if current_cluster is None or not cluster_cells[current_cluster]:
                best_cluster = None
                shortest_path_len = float('inf')
                for cluster in range(num_clusters):
                    if not cluster_cells[cluster]:
                        continue
                    rep_cell = get_cluster_representative(cluster_cells[cluster], rat_probs)
                    if rep_cell is None:
                        continue
                    path = bfs_path(ship, bot_pos, rep_cell)
                    if path and len(path) < shortest_path_len:
                        shortest_path_len = len(path)
                        best_cluster = cluster
                if best_cluster is None:
                    break
                current_cluster = best_cluster

            # Within the cluster, use cost minimization to select the target
            cluster_candidates = cluster_cells[current_cluster]
            if not cluster_candidates:
                current_cluster = None
                continue

            best_target = None
            min_expected_cost = float('inf')
            for target in cluster_candidates:
                path_to_target = bfs_path(ship, bot_pos, target)
                if not path_to_target:
                    continue
                expected_cost = 0.0
                for pos in cluster_candidates:
                    dist = manhattan_distance(target, pos)
                    expected_cost += rat_probs[pos] * dist
                if expected_cost < min_expected_cost:
                    min_expected_cost = expected_cost
                    best_target = target
                    target_path = path_to_target

            if not best_target:
                for target in cluster_candidates[:]:
                    del rat_probs[target]
                    rat_knowledge.remove(target)
                continue

        direction = target_path.pop(0)
        bot_pos = move(ship, bot_pos, direction)
        moves += 1

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)

        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0
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

        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.00001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

    return moves, senses, pings, estimated_spawn, true_rat_pos, caught


# ----- Comparison Logic -----
def evaluate_bots(ship_size=30, num_trials=250):
    """
    Evaluates both the baseline and area segmentation bots across alpha values.
    Returns average moves and success rates for both bots.
    """
    alpha_values = np.arange(0, 0.55, 0.05)  # 0, 0.05, 0.1, ..., 0.5
    baseline_avg_moves = []
    baseline_success_rates = []
    custom_avg_moves = []
    custom_success_rates = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        # Evaluate baseline bot
        baseline_moves = []
        baseline_successes = 0
        for _ in range(num_trials):
            result = baseline_bot(ship, alpha=alpha, verbose=False)
            moves = result[0]
            caught = result[-1]  # Last element is the caught flag
            baseline_moves.append(moves)
            if caught:
                baseline_successes += 1
        baseline_avg_moves.append(np.mean(baseline_moves))
        baseline_success_rates.append((baseline_successes / num_trials) * 100)
        print(f"Baseline: Alpha {alpha:.2f} done")

        # Evaluate custom bot (area segmentation)
        custom_moves = []
        custom_successes = 0
        for _ in range(num_trials):
            result = custom_bot_area_segmentation(ship, alpha=alpha, verbose=False)
            moves = result[0]
            caught = result[-1]
            custom_moves.append(moves)
            if caught:
                custom_successes += 1
        custom_avg_moves.append(np.mean(custom_moves))
        custom_success_rates.append((custom_successes / num_trials) * 100)
        print(f"Custom (Area Segmentation): Alpha {alpha:.2f} done")

    return alpha_values, baseline_avg_moves, baseline_success_rates, custom_avg_moves, custom_success_rates


def plot_comparison(alpha_values, baseline_avg_moves, baseline_success_rates, custom_avg_moves, custom_success_rates,
                    ship_size, num_trials):
    """
    Plots average moves and success rate vs alpha for both bots in a single graph.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot average moves on the left y-axis
    ax1.plot(alpha_values, baseline_avg_moves, marker='o', label='Baseline Bot (Moves)', color='skyblue')
    ax1.plot(alpha_values, custom_avg_moves, marker='o', label='Custom Bot (Moves)', color='salmon')
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Average Number of Moves", color='black')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Create a second y-axis for success rate
    ax2 = ax1.twinx()
    ax2.plot(alpha_values, baseline_success_rates, marker='s', label='Baseline Bot (Success Rate)', color='lightblue',
             linestyle='--')
    ax2.plot(alpha_values, custom_success_rates, marker='s', label='Custom Bot (Success Rate)', color='lightsalmon',
             linestyle='--')
    ax2.set_ylabel("Success Rate (%)", color='black')
    ax2.legend(loc='upper right')

    plt.title(
        f"Baseline vs Custom Bot: Moves and Success Rate vs Alpha\n(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.show()


if __name__ == "__main__":
    ship_size = 30
    num_trials = 250  # Matching your reduced trial count
    alpha_values, baseline_avg_moves, baseline_success_rates, custom_avg_moves, custom_success_rates = evaluate_bots(
        ship_size=ship_size, num_trials=num_trials)
    plot_comparison(alpha_values, baseline_avg_moves, baseline_success_rates, custom_avg_moves, custom_success_rates,
                    ship_size, num_trials)