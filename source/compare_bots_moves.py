import matplotlib.pyplot as plt
import numpy as np
import random
from ship_generator import generate_ship
from baseline_bot import baseline_bot  # Baseline bot
from custom_bot2 import unified_bot_stationary  # Custom bot with triangulation


def run_bots_same_ship(ship, alpha, num_trials=100):
    """
    Runs both bots on the same ship with the same random state for num_trials.
    Returns the average moves for baseline and unified bots.
    """
    baseline_moves = []
    unified_moves = []

    for _ in range(num_trials):
        # Capture the current random state for reproducibility
        state = random.getstate()

        # Run the baseline bot (stationary rat, no moving_rat parameter)
        result_baseline = baseline_bot(ship, alpha=alpha)
        moves_baseline = result_baseline[0]
        baseline_moves.append(moves_baseline)

        # Reset random state for the unified bot
        random.setstate(state)

        # Run the custom bot (unified bot with triangulation, stationary rat)
        result_unified = unified_bot_stationary(ship, alpha=alpha, max_steps_phase2=1000, replan_interval=3,
                                                triangulation_interval=5, sense_repeats=5)
        moves_unified = result_unified[0]
        unified_moves.append(moves_unified)

    avg_baseline = np.mean(baseline_moves)
    avg_unified = np.mean(unified_moves)
    return avg_baseline, avg_unified


def evaluate_bots_alpha(ship_size=30, num_trials=100):
    """
    Evaluates both bots across a range of alpha values and returns the average moves.
    """
    alpha_values = np.arange(0, 0.55, 0.05)  # 0, 0.05, 0.1, ..., 0.5
    baseline_avg_moves = []
    unified_avg_moves = []

    # Generate one ship for all runs
    ship = generate_ship(ship_size)

    for alpha in alpha_values:
        avg_baseline, avg_unified = run_bots_same_ship(ship, alpha, num_trials)
        baseline_avg_moves.append(avg_baseline)
        unified_avg_moves.append(avg_unified)

    return alpha_values, baseline_avg_moves, unified_avg_moves


def plot_moves_vs_alpha(alpha_values, baseline_moves, unified_moves, ship_size, num_trials):
    """
    Plots average moves vs alpha for both bots.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_values, baseline_moves, marker='o', label='Baseline Bot', color='skyblue')
    plt.plot(alpha_values, unified_moves, marker='o', label='Custom Bot (Triangulation)', color='salmon')
    plt.xlabel("Alpha")
    plt.ylabel("Average Number of Moves")
    plt.title(f"Average Moves vs Alpha\n(Ship: {ship_size}x{ship_size}, {num_trials} Trials per Alpha)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ship_size = 30
    num_trials = 100
    alpha_values, baseline_avg_moves, unified_avg_moves = evaluate_bots_alpha(ship_size=ship_size,
                                                                              num_trials=num_trials)
    plot_moves_vs_alpha(alpha_values, baseline_avg_moves, unified_avg_moves, ship_size, num_trials)