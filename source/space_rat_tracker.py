import numpy as np
from utils import manhattan_distance


def update_rat_knowledge(ship, bot_pos, ping, alpha, prev_probs=None):
    D = ship.shape[0]
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1]

    if prev_probs is None:
        prob = 1 / len(open_cells)
        probs = {pos: prob for pos in open_cells}
    else:
        probs = prev_probs.copy()

    for pos in probs:
        d = manhattan_distance(bot_pos, pos)
        likelihood = 1.0 if d == 0 and ping else np.exp(-alpha * (d - 1)) if ping else (
            1 - np.exp(-alpha * (d - 1)) if d > 0 else 0)
        probs[pos] *= likelihood

    total = sum(probs.values())
    if total > 0:
        for pos in probs:
            probs[pos] /= total

    return probs