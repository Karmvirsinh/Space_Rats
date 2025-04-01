import random
import numpy as np

def generate_ship(size):
    """
    Generates a spaceship grid layout with blocked edges and navigable paths.
    :param size: Size of the grid (e.g., 30 for a 30x30 grid).
    :return: The generated grid as a NumPy array (0 = blocked, 1 = open).
    """
    # Initialize grid with all cells blocked
    grid = [[0 for _ in range(size)] for _ in range(size)]

    # Ensure all edges are blocked
    for i in range(size):
        grid[0][i] = 0
        grid[size-1][i] = 0
        grid[i][0] = 0
        grid[i][size-1] = 0

    # Start with a random interior open cell
    start_x, start_y = random.randint(1, size-2), random.randint(1, size-2)
    grid[start_x][start_y] = 1

    # Function to get valid neighbors
    def get_neighbors(x, y):
        directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [(nx, ny) for nx, ny in directions if 1 <= nx < size-1 and 1 <= ny < size-1]

    # Generate maze using randomized approach
    frontier = [(start_x, start_y)]
    while frontier:
        current = random.choice(frontier)
        x, y = current
        neighbors = get_neighbors(x, y)

        # Limit open neighbors to avoid loops
        open_neighbors = [(nx, ny) for nx, ny in neighbors if grid[nx][ny] == 1]
        if len(open_neighbors) < 2:
            grid[x][y] = 1
            for nx, ny in neighbors:
                if grid[nx][ny] == 0:
                    frontier.append((nx, ny))
        frontier.remove(current)

    return np.array(grid)  # Convert to NumPy array

if __name__ == "__main__":
    grid = generate_ship(30)
    print(f"Generated {grid.shape[0]}x{grid.shape[1]} ship with {np.sum(grid)} open cells")