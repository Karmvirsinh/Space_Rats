import numpy as np
import random
import math
from collections import deque

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def ping_detector(bot_pos, rat_pos, alpha):
    """
    Simulate space rat detector ping.
    Probability = e^(-alpha * (distance - 1)) if distance > 0, else 1.
    :param bot_pos: Tuple (row, col) of bot position.
    :param rat_pos: Tuple (row, col) of rat position.
    :param alpha: Detector sensitivity (> 0).
    :return: True if there is ping, False if no ping.
    """
    distance = manhattan_distance(bot_pos, rat_pos)
    if distance == 0:
        return True  # Definitive ping if same cell
    prob = math.exp(-alpha * (distance - 1))
    return random.random() < prob

def sense_blocked(ship, pos):
    """
    Count blocked neighbors out of 8 surrounding cells.
    :param ship: 2D numpy array (1 = open, 0 = blocked).
    :param pos: Tuple (row, col) of current position.
    :return: Number of blocked neighbors.
    """
    row, col = pos
    D = ship.shape[0]
    neighbors = [
        (row-1, col-1), (row-1, col), (row-1, col+1),
        (row, col-1),                 (row, col+1),
        (row+1, col-1), (row+1, col), (row+1, col+1)
    ]
    blocked = sum(1 for r, c in neighbors if 0 <= r < D and 0 <= c < D and ship[r, c] == 0)
    return blocked

def move(ship, pos, direction):
    """
    Attempt to move in a cardinal direction.
    :param ship: 2D numpy array (1 = open, 0 = blocked).
    :param pos: Tuple (row, col) of current position.
    :param direction: String ('up', 'down', 'left', 'right').
    :return: New position if move succeeds, current position if blocked.
    """
    row, col = pos
    D = ship.shape[0]
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    delta_row, delta_col = moves.get(direction, (0, 0))
    new_row, new_col = row + delta_row, col + delta_col
    if 0 <= new_row < D and 0 <= new_col < D and ship[new_row, new_col] == 1:
        return (new_row, new_col)
    return pos

def bfs_path(ship, start, target):
    """
    Find a path from start to target using BFS.
    Args:
        ship: 2D numpy array representing the maze (1 for open, 0 for walls).
        start: Tuple (r, c) of starting position.
        target: Tuple (r, c) of target position.
    Returns:
        List of directions ['up', 'down', 'left', 'right'] to follow, or None if no path.
    """
    if start == target:
        return []

    D = ship.shape[0]
    directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        for direction, (dr, dc) in directions.items():
            nr, nc = r + dr, c + dc
            if (nr, nc) == target:
                return path + [direction]
            if (0 <= nr < D and 0 <= nc < D and ship[nr, nc] == 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [direction]))

    return None  # No path found