import numpy as np  
from utils import manhattan_distance  


def update_rat_knowledge(ship, bot_pos, ping, alpha, prev_probs=None):
    """
    Updates our belief (knowledge) about where the rat could be on the ship.
    
    Parameters:
      ship: The grid (map) of the ship, where 1 means open cell and 0 means blocked.
      bot_pos: The current position (cell) of our bot.
      ping: A flag (True or False) indicating whether the bot's sensor (ping) detected the rat.
      alpha: A parameter that controls how quickly the chance of detection decreases with distance.
      prev_probs: (Optional) The previous probabilities for each cell being the rat's location.
    
    Returns:
      A dictionary (mapping) where each open cell is given a probability representing how likely
      it is to have the rat.
    """
    # Get the size (length) of the grid.
    D = ship.shape[0]
    # Create a list of all cells in the grid that are open (have a value of 1).
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1]

    # If no previous probabilities are given, we start with an equal chance for all open cells.
    if prev_probs is None:
        prob = 1 / len(open_cells)  # Each open cell has the same initial probability.
        probs = {pos: prob for pos in open_cells}  # Create a dictionary with these probabilities.
    else:
        # Otherwise, we work with the previous probabilities we have.
        probs = prev_probs.copy()

    # Now, update the probability for each cell based on how far it is from the bot and whether a ping was heard.
    for pos in probs:
        # Calculate the Manhattan distance from the bot's position to the current cell.
        d = manhattan_distance(bot_pos, pos)
        # Calculate how likely it is for the rat to be in this cell:
        # - If the cell is the bot's own position (distance 0) and a ping is received, the likelihood is 1.
        # - If a ping is heard for a cell that's not at the bot's position, we use an exponential decay function.
        # - If no ping is heard, then the likelihood is lower for cells farther away.
        likelihood = 1.0 if d == 0 and ping else np.exp(-alpha * (d - 1)) if ping else (
            1 - np.exp(-alpha * (d - 1)) if d > 0 else 0)
        # Multiply the current probability by this likelihood to update it.
        probs[pos] *= likelihood

    # After updating, we need to normalize the probabilities so they add up to 1.
    total = sum(probs.values())
    if total > 0:
        for pos in probs:
            probs[pos] /= total  # Divide each probability by the total sum.

    # Return the updated probabilities dictionary.
    return probs
