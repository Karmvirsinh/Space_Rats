# custom_bot_triangulation.py
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
def argmax_candidate(prob_dict):
    return max(prob_dict, key=prob_dict.get)


# ----- Custom Bot with Improved Triangulation (Stationary Rat) -----
def custom_bot_triangulation(ship, alpha=0.15, verbose=False, triangulation_interval=3, sense_repeats=10):
    D = ship.shape[0]
    moves = 0
    senses = 0
    pings = 0

    # --- Phase 1: Localization ---
    candidate_set = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(candidate_set))
    move_history = []
    current_sensor = get_surroundings(ship, true_bot_pos)
    senses += 1
    candidate_set = {pos for pos in candidate_set if get_surroundings(ship, pos) == current_sensor}

    while len(candidate_set) > 1:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, true_bot_pos, m) != true_bot_pos]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)
        true_bot_pos = move(ship, true_bot_pos, chosen_move)
        moves += 1
        new_sensor = get_surroundings(ship, true_bot_pos)
        senses += 1
        new_candidate_set = set()
        for pos in candidate_set:
            new_pos = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos) == new_sensor:
                new_candidate_set.add(new_pos)
        candidate_set = new_candidate_set

    if len(candidate_set) == 1:
        unique_candidate = candidate_set.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
    else:
        estimated_spawn = true_bot_pos

    bot_pos = estimated_spawn

    # --- Phase 2: Rat Tracking (Stationary Rat with Improved Triangulation) ---
    rat_candidates = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != bot_pos}
    true_rat_pos = random.choice(list(rat_candidates))
    rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}
    max_steps = 1000
    steps_since_triangulation = 0

    while moves < max_steps:
        # Triangulation every triangulation_interval steps
        if steps_since_triangulation >= triangulation_interval:
            # Sense multiple times to estimate rat distance
            beeps = 0
            for _ in range(sense_repeats):
                sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
                pings += 1
                if sensor_ping:
                    beeps += 1
            beep_rate = beeps / sense_repeats
            # Estimate distance d: e^(-alpha*(d-1)) ≈ beep_rate
            if 0 < beep_rate < 1:  # Avoid log(0) or log(1)
                est_dist = 1 - np.log(beep_rate) / alpha
                est_dist = max(1, round(est_dist))  # Ensure distance is at least 1
            else:
                est_dist = 1 if beep_rate == 1 else 10  # Fallback: close if always beeps, far if never beeps
            # Adjust probabilities: boost cells around the estimated distance
            new_probs = {}
            total_prob = 0.0
            for pos, prob in rat_probs.items():
                d = manhattan_distance(bot_pos, pos)
                if d == 0:
                    new_probs[pos] = 0.0  # Rat not at bot's position
                else:
                    # Tighter Gaussian weighting: higher probability if d is close to est_dist
                    weight = np.exp(-0.5 * ((d - est_dist) ** 2) / 1)  # Std dev reduced to 1
                    new_p = prob * weight
                    new_probs[pos] = new_p
                    total_prob += new_p
            if total_prob > 0:
                for pos in new_probs:
                    new_probs[pos] /= total_prob
            rat_probs = new_probs
            steps_since_triangulation = 0
        else:
            # Check if rat is caught before moving
            if manhattan_distance(bot_pos, true_rat_pos) == 0:
                sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
                pings += 1
                if sensor_ping:
                    break

            # Bayesian update
            sensor_ping = ping_detector(bot_pos, true_rat_pos, alpha)
            pings += 1
            new_rat_probs = {}
            total_prob = 0.0
            for pos, prob in rat_probs.items():
                d = manhattan_distance(bot_pos, pos)
                if d == 0:
                    likelihood = 1.0
                else:
                    base = np.exp(-alpha * (d - 1))
                    likelihood = base if sensor_ping else (1.0 - base)
                new_p = prob * likelihood
                new_rat_probs[pos] = new_p
                total_prob += new_p
            if total_prob > 0:
                for pos in new_rat_probs:
                    new_rat_probs[pos] /= total_prob
            rat_probs = new_rat_probs

        # Prune low-probability cells
        threshold = 0.0001
        rat_probs = {pos: prob for pos, prob in rat_probs.items() if prob >= threshold}
        total_prob = sum(rat_probs.values())
        if total_prob > 0:
            rat_probs = {pos: prob / total_prob for pos, prob in rat_probs.items()}
        else:
            rat_probs = {pos: 1.0 / len(rat_candidates) for pos in rat_candidates}

        # Move toward the highest-probability cell
        target = argmax_candidate(rat_probs)
        path = bfs_path(ship, bot_pos, target)
        if not path:
            valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
            if not valid_moves:
                break
            chosen_move = random.choice(valid_moves)
        else:
            chosen_move = path[0]
        bot_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        steps_since_triangulation += 1

    return moves, senses, pings, estimated_spawn, true_rat_pos


# ----- Plotting Logic (Only for Custom Bot) -----
def evaluate_custom_bot_alpha(ship_size=30, num_trials=100):
    """
    Evaluates the custom bot across a range of alpha values and returns the average moves.
    """
    alpha_values = np.arange(0, 0.55, 0.05)  # 0, 0.05, 0.1, ..., 0.5
    custom_avg_moves = []

    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        custom_moves = []
        for _ in range(num_trials):
            result_custom = custom_bot_triangulation(ship, alpha=alpha, verbose=False)
            moves_custom = result_custom[0]
            custom_moves.append(moves_custom)
        avg_custom = np.mean(custom_moves)
        custom_avg_moves.append(avg_custom)

    return alpha_values, custom_avg_moves


def plot_moves_vs_alpha(alpha_values, custom_moves, ship_size, num_trials):
    """
    Plots average moves vs alpha for the custom bot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_values, custom_moves, marker='o', label='Custom Bot (Triangulation)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.title(f"Average Moves vs Alpha (Custom Bot)\n(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    alpha_values, custom_avg_moves = evaluate_custom_bot_alpha(ship_size=ship_size, num_trials=num_trials)
    plot_moves_vs_alpha(alpha_values, custom_avg_moves, ship_size, num_trials)