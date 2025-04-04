import numpy as np
import matplotlib.pyplot as plt
from baseline_bot1_withratmove import baseline_bot
from ship_generator import generate_ship

def visualize_bot(ship):
    moves, senses, pings, steps, true_rat_pos = baseline_bot(ship, alpha=0.15)

    plt.ion()
    fig, ax = plt.subplots()

    phase_2_start_step = None
    for i, (bot_pos, m, s, p, knowledge_size, rat_pos) in enumerate(steps):
        if p > 0:
            phase_2_start_step = i
            break
    if phase_2_start_step is None:
        print("Phase 2 never started (no pings).")
        return

    moving_rat = any(steps[i][5] != steps[i+1][5] for i in range(phase_2_start_step, len(steps)-1) if steps[i][5] is not None and steps[i+1][5] is not None)

    for i in range(phase_2_start_step, len(steps)):
        bot_pos, m, s, p, knowledge_size, current_rat_pos = steps[i]
        if current_rat_pos is None:
            current_rat_pos = true_rat_pos

        grid = np.ones((ship.shape[0], ship.shape[0], 3))
        grid[ship == 0] = [0, 0, 0]
        grid[bot_pos] = [1, 0, 0]
        grid[current_rat_pos] = [0, 1, 0]
        if bot_pos == current_rat_pos:
            grid[bot_pos] = [1, 0, 0]

        ax.clear()
        ax.imshow(grid)
        ax.set_title(f"Phase 2: {'Moving' if moving_rat else 'Stationary'} Rat\nStep {i}: Bot Pos {bot_pos}, Rat Pos {current_rat_pos}\nMoves: {m}, Senses: {s}, Pings: {p}, Rat KB: {knowledge_size}")
        plt.pause(0.5)

    final_pos, m, s, p, knowledge_size, final_rat_pos = steps[-1]
    grid = np.ones((ship.shape[0], ship.shape[0], 3))
    grid[ship == 0] = [0, 0, 0]
    grid[final_pos] = [1, 0, 0]
    grid[final_rat_pos] = [0, 1, 0]
    if final_pos == final_rat_pos:
        grid[final_pos] = [1, 0, 0]
    ax.clear()
    ax.imshow(grid)
    ax.set_title(f"Phase 2 Done - {'Moving' if moving_rat else 'Stationary'} Rat\nBot at {final_pos}, Rat at {final_rat_pos}\nMoves: {moves}, Senses: {senses}, Pings: {pings}")
    plt.pause(2)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    ship = generate_ship(30)
    visualize_bot(ship)