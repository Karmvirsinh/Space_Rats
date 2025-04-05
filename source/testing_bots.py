import matplotlib.pyplot as plt
import numpy as np
import random
from ship_generator import generate_ship
from baseline_bot import baseline_bot           # Adjust this import if needed.
from custom_bot_me import unified_bot_stationary  # Adjust this import if needed.

def run_bots_same_ship(ship_size=30, alpha=0.15):
    """
    Generates one ship, then runs both bots (baseline and unified stationary) on that same ship and 
    with the same random state so that they get the same bot and rat spawn positions.
    Returns a tuple (moves_baseline, moves_unified).
    """
    # Generate one ship.
    ship = generate_ship(ship_size)
    # Capture the current random state.
    state = random.getstate()
    
    # Run the baseline bot on this ship.
    result_baseline = baseline_bot(ship, alpha=alpha)
    moves_baseline = result_baseline[0]
    
    # Reset random state so the unified bot sees the same random sequence.
    random.setstate(state)
    
    # Run the improved unified bot (stationary rat version) on the same ship.
    result_unified = unified_bot_stationary(ship, alpha=alpha, max_steps_phase2=1000)
    moves_unified = result_unified[0]
    
    return moves_baseline, moves_unified

def test_bots(num_runs=100, ship_size=30, alpha=0.15):
    baseline_moves = []
    unified_moves = []
    for i in range(num_runs):
        mb, mu = run_bots_same_ship(ship_size=ship_size, alpha=alpha)
        baseline_moves.append(mb)
        unified_moves.append(mu)
        print(f"Run {i+1:3d}: Baseline moves = {mb}, Unified moves = {mu}")
    avg_baseline = np.mean(baseline_moves)
    avg_unified = np.mean(unified_moves)
    return avg_baseline, avg_unified, baseline_moves, unified_moves

def main():
    num_runs = 100
    ship_size = 30
    alpha = 0.15
    avg_baseline, avg_unified, baseline_moves, unified_moves = test_bots(num_runs=num_runs, ship_size=ship_size, alpha=alpha)
    print(f"\nAverage moves (Baseline Bot): {avg_baseline:.2f}")
    print(f"Average moves (Unified Bot): {avg_unified:.2f}")
    
    # Plot the average moves as a bar graph.
    bots = ['Baseline Bot', 'Unified Bot']
    averages = [avg_baseline, avg_unified]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(bots, averages, color=['skyblue', 'salmon'])
    plt.ylabel("Average Number of Moves")
    plt.title(f"Average Moves over {num_runs} Runs\nShip: {ship_size}x{ship_size}, alpha={alpha}")
    for bar, avg in zip(bars, averages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{avg:.1f}", ha="center", va="bottom")
    plt.ylim(0, max(averages) * 1.2)
    plt.show()

if __name__ == "__main__":
    main()
