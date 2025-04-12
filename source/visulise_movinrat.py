import numpy as np                   
import matplotlib.pyplot as plt      
from custom_bot_bayesian import custom_bayesian_bot  
from ship_generator import generate_ship  


def visualize_bot(ship):
    """
    This function runs the smart bot (custom_bayesian_bot) on the ship grid 
    and shows a picture of what the bot sees over time.
    The picture updates to show the bot's and the rat's positions as the bot moves.
    """
    # Run the custom Bayesian bot with a parameter alpha of 0.15.
    # The bot returns the number of moves, sensor readings, pings, a list of all steps, 
    # and the true final position of the rat.
    moves, senses, pings, steps, true_rat_pos = custom_bayesian_bot(ship, alpha=0.15)

    # Turn on interactive mode so we can update the plot step-by-step.
    plt.ion()
    # Create a figure and an axis object where our grid will be drawn.
    fig, ax = plt.subplots()

    # Find the step when Phase 2 starts by looking for the first step 
    # where the bot has performed a "ping" (a check signal).
    phase_2_start_step = None
    for i, (bot_pos, m, s, p, knowledge_size, rat_pos) in enumerate(steps):
        if p > 0:  # A ping has occurred, which means Phase 2 (tracking) has started.
            phase_2_start_step = i
            break
    if phase_2_start_step is None:
        print("Phase 2 never started (no pings).")
        return

    # Check if the rat is moving by comparing the rat's position from one step to the next.
    moving_rat = any(
        steps[i][5] != steps[i+1][5] 
        for i in range(phase_2_start_step, len(steps) - 1)
        if steps[i][5] is not None and steps[i+1][5] is not None
    )

    # Loop through each step, starting from when the bot began tracking the rat.
    for i in range(phase_2_start_step, len(steps)):
        # Unpack details for each step: bot position, move count, sensor count, ping count, 
        # size of the bot's knowledge about the rat, and the rat's current (or estimated) position.
        bot_pos, m, s, p, knowledge_size, current_rat_pos = steps[i]
        # If we didn't capture a rat position at this step, use the true rat position.
        if current_rat_pos is None:
            current_rat_pos = true_rat_pos

        # Create a grid picture where every cell is initially white.
        # The grid is 3-dimensional because we are using RGB colors.
        grid = np.ones((ship.shape[0], ship.shape[0], 3))
        # Set blocked cells (walls) to black.
        grid[ship == 0] = [0, 0, 0]
        # Mark the bot's position with red.
        grid[bot_pos] = [1, 0, 0]
        # Mark the rat's current position with green.
        grid[current_rat_pos] = [0, 1, 0]
        # If the bot's position and the rat's position are the same,
        # then the bot's color (red) remains visible.
        if bot_pos == current_rat_pos:
            grid[bot_pos] = [1, 0, 0]

        # Clear any previous drawing on the plot.
        ax.clear()
        # Show the current grid image.
        ax.imshow(grid)
        # Set the title of the plot to display current status: 
        # whether the rat is moving or stationary, current step, positions, and counts.
        ax.set_title(
            f"Phase 2: {'Moving' if moving_rat else 'Stationary'} Rat\n"
            f"Step {i}: Bot Pos {bot_pos}, Rat Pos {current_rat_pos}\n"
            f"Moves: {m}, Senses: {s}, Pings: {p}, Rat KB: {knowledge_size}"
        )
        # Pause for a short moment (0.5 seconds) before moving to the next step.
        plt.pause(0.5)

    # After the loop ends, show the final positions.
    final_pos, m, s, p, knowledge_size, final_rat_pos = steps[-1]
    grid = np.ones((ship.shape[0], ship.shape[0], 3))
    grid[ship == 0] = [0, 0, 0]
    grid[final_pos] = [1, 0, 0]         # Bot's final position in red.
    grid[final_rat_pos] = [0, 1, 0]       # Rat's final position in green.
    if final_pos == final_rat_pos:
        grid[final_pos] = [1, 0, 0]
    ax.clear()
    ax.imshow(grid)
    # Set a final title showing that Phase 2 is done and displaying summary information.
    ax.set_title(
        f"Phase 2 Done - {'Moving' if moving_rat else 'Stationary'} Rat\n"
        f"Bot at {final_pos}, Rat at {final_rat_pos}\n"
        f"Moves: {moves}, Senses: {senses}, Pings: {pings}"
    )
    # Pause for 2 seconds so the final result can be seen.
    plt.pause(2)

    # Turn off interactive mode and show the final window permanently.
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Generate a ship (grid) of size 30x30 with blocked and open cells.
    ship = generate_ship(30)
    # Run the visualization function to see how the bot tracks the rat.
    visualize_bot(ship)
