import numpy as np
import matplotlib.pyplot as plt
from baseline_bot import baseline_bot
from ship_generator import generate_ship

def visualize_bot(ship):
    # Run the bot and get true_rat_pos
    moves, senses, pings, steps, true_rat_pos = baseline_bot(ship)

    plt.ion()
    fig, ax = plt.subplots()

    # Find the start of Phase 2
    phase_2_start_step = None
    for i, (bot_pos, m, s, p, knowledge_size) in enumerate(steps):
        # Phase 2 starts when pings > 0 (first ping indicates Phase 2)
        if p > 0:
            phase_2_start_step = i
            break
    if phase_2_start_step is None:
        print("Phase 2 never started (no pings).")
        return

    # Visualize only Phase 2 steps
    for i in range(phase_2_start_step, len(steps)):
        bot_pos, m, s, p, knowledge_size = steps[i]

        # Create the grid for visualization
        grid = np.ones((ship.shape[0], ship.shape[0], 3))
        grid[ship == 0] = [0, 0, 0]  # Walls: black
        grid[bot_pos] = [1, 0, 0]     # Bot: red
        grid[true_rat_pos] = [0, 1, 0]  # Rat: green
        # If bot and rat are in the same cell, show bot (red) on top
        if bot_pos == true_rat_pos:
            grid[bot_pos] = [1, 0, 0]

        ax.clear()
        ax.imshow(grid)
        ax.set_title(f"Phase 2: Finding Rat\nStep {i}: Bot Pos {bot_pos}, Moves: {m}, Senses: {s}, Pings: {p}, Rat KB: {knowledge_size}")
        plt.pause(0.5)

    # Final frame
    final_pos = steps[-1][0]
    grid[final_pos] = [1, 0, 0]
    grid[true_rat_pos] = [0, 1, 0]
    if final_pos == true_rat_pos:
        grid[final_pos] = [1, 0, 0]  # Bot on top if they overlap
    ax.clear()
    ax.imshow(grid)
    ax.set_title(f"Phase 2 Done - Bot at {final_pos}, Rat at {true_rat_pos}\nMoves: {moves}, Senses: {senses}, Pings: {pings}")
    plt.pause(2)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    ship = generate_ship(30)
    visualize_bot(ship)