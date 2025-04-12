import random  
import numpy as np  

def generate_ship(size):
    """
    Generates a spaceship grid layout with blocked edges and navigable paths.
    
    The function produces a maze-like grid represented by a 2D NumPy array where:
      - 0 indicates a blocked cell.
      - 1 indicates an open (navigable) cell.
      
    The grid's outer edges are always blocked, and the interior is carved out using
    a randomized maze generation algorithm.
    
    Parameters:
      - size: Integer defining the dimensions of the grid (size x size). For example, size=30 yields a 30x30 grid.
    
    Returns:
      - A NumPy array representing the generated ship grid.
    """
    # Initialize the grid as a list of lists filled with 0s (all cells blocked).
    grid = [[0 for _ in range(size)] for _ in range(size)]

    # Ensure that all boundary cells (edges) remain blocked.
    for i in range(size):
        grid[0][i] = 0           # Top edge of the grid.
        grid[size-1][i] = 0        # Bottom edge of the grid.
        grid[i][0] = 0           # Left edge of the grid.
        grid[i][size-1] = 0        # Right edge of the grid.

    # Select a random cell within the interior (avoid borders) as the starting point.
    start_x, start_y = random.randint(1, size-2), random.randint(1, size-2)
    grid[start_x][start_y] = 1  # Mark the starting cell as open.

    # Nested function to get valid neighbor cells within the interior.
    def get_neighbors(x, y):
        # List possible neighbors (up, down, left, right).
        directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        # Return only those neighbor cells that are strictly within the interior boundaries.
        return [(nx, ny) for nx, ny in directions if 1 <= nx < size-1 and 1 <= ny < size-1]

    # Begin maze generation using a randomized approach.
    # The "frontier" holds cells that are candidates for expansion.
    frontier = [(start_x, start_y)]
    while frontier:
        # Select a random cell from the frontier.
        current = random.choice(frontier)
        x, y = current
        # Retrieve valid neighbors for the current cell.
        neighbors = get_neighbors(x, y)

        # Check how many of the neighboring cells are already open.
        # This is to avoid creating loops by not opening more than one neighbor excessively.
        open_neighbors = [(nx, ny) for nx, ny in neighbors if grid[nx][ny] == 1]
        if len(open_neighbors) < 2:
            # If not too many open neighbors, mark the current cell as open.
            grid[x][y] = 1
            # Add neighbor cells that are still blocked to the frontier to consider later.
            for nx, ny in neighbors:
                if grid[nx][ny] == 0:
                    frontier.append((nx, ny))
        # Remove the current cell from the frontier after processing it.
        frontier.remove(current)

    # Convert the grid (list of lists) into a NumPy array before returning it.
    return np.array(grid)

if __name__ == "__main__":
    # When run as the main module, generate a ship grid of size 30x30.
    grid = generate_ship(30)
    # Print out the grid dimensions and the number of open cells in the generated ship.
    print(f"Generated {grid.shape[0]}x{grid.shape[1]} ship with {np.sum(grid)} open cells")
