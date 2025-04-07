# custom_bot_area_segmentation.py
import numpy as np
import random
from ship_generator import generate_ship
from utils import move, ping_detector, manhattan_distance, bfs_path

# ----- Helper Functions for Phase 1 (Localization) -----
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

# ----- Helper Functions for Phase 2 (Rat Tracking) -----
def get_cluster(pos, D, cluster_size):
    """Assign a cell to a cluster based on its position (grid-based clustering)."""
    r, c = pos
    cluster_row = r // cluster_size
    cluster_col = c // cluster_size
    clusters_per_row = D // cluster_size
    return cluster_row * clusters_per_row + cluster_col

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
    max_steps = 1000
    target_path = []
    current_cluster = None

    # Dynamic clustering: divide the ship into a grid of clusters
    cluster_size = 10  # Each cluster is 10x10 (for a 30x30 ship, this gives 9 clusters)
    num_clusters = (D // cluster_size) ** 2

    while moves < max_steps:
        if manhattan_distance(bot_pos, true_rat_pos) == 0:
            sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
            pings += 1
            if sensor_ping:
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

            # Choose the cluster with the highest total probability
            if current_cluster is None or not cluster_cells[current_cluster]:
                current_cluster = np.argmax(cluster_probs)
                if cluster_probs[current_cluster] == 0:
                    break  # No more cells to explore

            # Within the cluster, use cost minimization to select the target
            cluster_candidates = cluster_cells[current_cluster]
            if not cluster_candidates:
                current_cluster = None  # Move to a new cluster
                continue

            best_target = None
            min_expected_cost = float('inf')
            for target in cluster_candidates:
                path_to_target = bfs_path(ship, bot_pos, target)
                if not path_to_target:
                    continue
                # Expected cost = sum(P(pos) * ManhattanDistance(target, pos)) within the cluster
                expected_cost = 0.0
                for pos in cluster_candidates:
                    dist = manhattan_distance(target, pos)
                    expected_cost += rat_probs[pos] * dist
                if expected_cost < min_expected_cost:
                    min_expected_cost = expected_cost
                    best_target = target
                    target_path = path_to_target

            if not best_target:
                # Fallback: remove unreachable cells and try again
                for target in cluster_candidates[:]:  # Copy to allow removal
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

        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

    return moves, senses, pings, estimated_spawn, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, estimated_spawn, true_rat_pos = custom_bot_area_segmentation(ship)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")